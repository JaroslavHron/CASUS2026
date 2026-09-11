from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
dt = opts.getReal('dt', 0.1)      # time step
t_end  = opts.getReal('t_end', 20.0)  # end time
lmbda  = opts.getReal('lambda', 0.02) # interface parameter
M_mob  = opts.getReal('M', 1.0)       # mobility

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler, RadauIIA, BDF

#scheme = BackwardEuler()
#scheme = ContinuousPetrovGalerkinScheme(2)
scheme = RadauIIA(2)
#scheme = BDF(2)

dt = Constant(dt)
t = Constant(0.0)

# Create mesh and define function spaces
n = 40
mesh = RectangleMesh(n, n, 1.0, 1.0)

k = 1
#V = VectorFunctionSpace(mesh, "CG", k+1)
#P = FunctionSpace(mesh, "CG", k)
C = FunctionSpace(mesh, "CG", k)
M = FunctionSpace(mesh, "CG", k)

# Alternative stable, pressure robust elements - Scott-Vogelius
V = VectorFunctionSpace(mesh, "CG", k+1, variant="alfeld")
P = FunctionSpace(mesh, "DG", k, variant="alfeld")

W = MixedFunctionSpace([V, P, C, M])

# Define test functions
v_, p_, c_, m_ = TestFunctions(W)

# Define functions
w = Function(W)  # current solution

# Split mixed functions
v, p, c, m = split(w)

# Boundary conditions
bcv_wall = DirichletBC(W.sub(0), as_vector([0, 0]), [1, 2, 3, 4])
bcs = [bcv_wall]

# Spatial coordinates for initial condition
x, y = SpatialCoordinate(mesh)

# Physical parameters
g = Constant([0.0, -1.0])  # nondimensional gravity vector

# Fluid properties (phase-dependent on concentration)
rho_1 = 1e2   # light phase
rho_2 = 1e3   # heavy phase
mu_1 = 1e1    # viscosity light
mu_2 = 1e0    # viscosity heavy
sigma = Constant(1.0)  # surface tension

# Concentration-dependent properties
def rho_c(c_val):
    return rho_1 + (rho_2 - rho_1) * c_val

def mu_c(c_val):
    return mu_1 + (mu_2 - mu_1) * c_val

# Cahn-Hilliard chemical potential
c_var = variable(c)
#f = 0.25 * (c_var**2 - 1)**2
f = c_var**2 * (1 - c_var**2)
dfdc = diff(f, c_var)

# Mesh-dependent parameter
eps = 2.0*CellDiameter(mesh)

# Initial level-set configuration
center = [0.5, 0.5]
radius = 0.2

bubble = sqrt((x - center[0])**2 + (y - center[1])**2) - radius
base = y - 0.2

def min_func(a, b):
    return conditional(a < b, a, b)

dist = min_func(base, bubble)

def Sign(q, eps):
    #return q / sqrt(q * q + eps * eps)
    return tanh(3.8 * q / eps)
    #condlist   = [q < -eps, (q >= -eps) & (q <= eps), q > eps]
    #choicelist = [ -1.0, q / eps, 1.0]
    #choicelist = [ -1.0, q / eps + np.sin(np.pi * q / eps) / np.pi), 1.0]
    return np.select(condlist, choicelist)

c_init = 0.5 * (1.0 - Sign(dist, eps))

# Strain rate and stress tensor
I = Identity(mesh.topological_dimension)
D = sym(grad(v))
T = -p * I + 2 * mu_c(c) * D

# Capillary force
f_cap = sigma * grad(c)

print(f"rho_ratio={rho_2/rho_1} mu_ratio={mu_2/mu_1}")

# Weak statement of the equations
# Cahn-Hilliard (mixed form):
#   Dt(c) + v·grad(c) - div(M*grad(m)) = 0
#   m - df/dc + lambda^2*div(grad(c)) = 0
# Navier-Stokes: rho(c)*Dt(v) + rho(c)*(grad(v)*v) - div(T) + grad(p) = rho(c)*g + f_cap
# Continuity: div(v) = 0

L_momentum = (
    rho_c(c) * inner(dot(grad(v), v), v_) * dx
    + inner(T, grad(v_)) * dx
    - rho_c(c) * inner(g, v_) * dx
    - inner(f_cap, v_) * dx
)

L_continuity = div(v) * p_ * dx

L_ch0 = Dt(c) * c_ * dx + inner(dot(grad(c), v), c_) * dx + M_mob * inner(grad(m), grad(c_)) * dx

L_ch1 = m * m_ * dx - dfdc * m_ * dx - lmbda**2 * inner(grad(c), grad(m_)) * dx

F = inner(rho_c(c) * Dt(v), v_) * dx + L_momentum + L_continuity + L_ch0 + L_ch1

nullsp = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm), W.sub(2), W.sub(3)])
stepper = TimeStepper(F, scheme, t, dt, w, bcs=bcs, options_prefix="", nullspace=nullsp)

# Output file
vtk = VTKFile("results/ch_nse_rising_bubble.pvd")

v, p, c, m = w.subfunctions
v.rename("v", "velocity")
p.rename("p", "pressure")
c.rename("c", "concentration")
m.rename("m", "chemical_potential")

# Initial conditions (zero everywhere, boundary conditions will be applied)
# Initialize concentration
c.interpolate(c_init)

# This is the default for Firedrake functions
c0 = Function(c.function_space())
c0.assign(c)

# Time stepping
T = Constant(t_end)

print(f"{float(t)=:4e}")
vtk.write(v, p, c, m, time=float(t))

while float(t) < float(T):
    stepper.advance()
    t.assign(t+dt)
    print(stepper.solver_stats())
    
    # Monitor bubble volume and mass
    V = assemble(conditional(c > 0.5, 1.0, 0.0) * dx)
    c_avg = assemble(c * dx)
    
    #dcdt = assemble(L_ch0)
    #for bc in bcs: bc.zero(dcdt)
    #dcdt = dcdt.riesz_representation()
    
    #ddcdt = (c-c0)/dt
    #print(f"{float(t)=:4e} volume={V:e} c_avg={c_avg:e} {norm(dcdt)=}  {norm(ddcdt)=}")
    c0.assign(c)
    
    vtk.write(v, p, c, m, time=float(t))
