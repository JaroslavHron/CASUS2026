from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
dt_val = opts.getReal('dt', 0.01)  # time step
t_end  = opts.getReal('t_end', 10.0)  # end time
lmbda  = opts.getReal('lambda', 1.0e-02)  # interface parameter (Cahn-Hilliard)
M_mob  = opts.getReal('M', 1.0)  # mobility (Cahn-Hilliard)

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler

scheme = BackwardEuler()

dt = Constant(dt_val)
t = Constant(0.0)

# Create mesh and define function spaces
n = 20
mesh = RectangleMesh(n, 2*n, 1.0, 2.0)

# Mixed element: velocity (CG2 vector), pressure (CG1), concentration (CG1)
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = VectorElement("CG", mesh.ufl_cell(), 2)
Ec = FiniteElement("CG", mesh.ufl_cell(), 1)

Evpc = MixedElement([Ev, Ep, Ec])
W = FunctionSpace(mesh, Evpc)

# Define test functions
v_, p_, c_ = TestFunctions(W)

# Define functions
w = Function(W)  # current solution

# Split mixed functions
v, p, c = split(w)

# Boundary conditions
bcv = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), [1, 2, 3, 4])
bcc_top = DirichletBC(W.sub(2), 1.0, [4])  # concentration = 1 at top
bcc_bot = DirichletBC(W.sub(2), 0.0, [3])  # concentration = 0 at bottom

bcs = [bcv, bcc_top, bcc_bot]

# Physical parameters
g = Constant([0.0, -1.0])  # nondimensional gravity vector

# Fluid properties (phase-dependent)
# Densities and viscosities interpolated from concentration
rho_1 = 1e2  # light phase (bubble)
rho_2 = 1e3  # heavy phase (fluid)
mu_1 = 1e1   # viscosity light
mu_2 = 1e0   # viscosity heavy
sigma = Constant(1.0)  # surface tension coefficient

# Smooth transition functions based on concentration c in [0,1]
def rho_c(c_val):
    """Density as function of concentration"""
    return rho_1 + (rho_2 - rho_1) * c_val


def mu_c(c_val):
    """Viscosity as function of concentration"""
    return mu_1 + (mu_2 - mu_1) * c_val


# Cahn-Hilliard chemical potential
c_var = variable(c)
f = 0.25 * (c_var**2 - 1)**2  # double-well potential
dfdc = diff(f, c_var)

# Strain rate and stress tensor
D = sym(grad(v))
T = 2 * mu_c(c) * D

# Mesh-dependent parameter for smooth transitions
eps = CellDiameter(mesh)

# Spatial coordinates for initial condition
x, y = SpatialCoordinate(mesh)

# Initial concentration (bubbles at top)
center = [0.5, 0.5]
radius = 0.25

bubble1 = sqrt((x - 0.5)**2 + (y - 1.5)**2) - 0.25
bubble2 = sqrt((x - 0.3)**2 + (y - 1.3)**2) - 0.20
base = y - 0.5


def min_func(a, b):
    return conditional(a < b, a, b)


dist = min_func(base, min_func(bubble1, bubble2))

# Initial concentration: 1 in bubbles (light phase), 0 in base fluid
c_init = 0.5 * (1.0 - tanh(10.0 * dist))

# Weak statement of the equations
# Cahn-Hilliard equation:
#   Dt(c) + v·grad(c) = div(M*grad(mu))
# where mu = df/dc - lambda^2 * Laplacian(c)
#
# Navier-Stokes with concentration-dependent properties:
#   rho(c)*Dt(v) + rho(c)*(grad(v)*v) - div(T) + grad(p) = rho(c)*g + capillary force
#   div(v) = 0
#
# Capillary force from Cahn-Hilliard: f_cap = -sigma * div(c * Identity - c * Identity)
# Simplified: proportional to grad(c) at interface

# Chemical potential
mu = dfdc - lmbda**2 * div(grad(c))

# Cahn-Hilliard transport (with convection by velocity field)
L_ch = Dt(c) * c_ * dx + inner(v, grad(c)) * c_ * dx + M_mob * inner(grad(mu), grad(c_)) * dx

# Capillary force contribution to momentum
# F_cap = sigma * grad(c) (simplified interfacial force)
f_cap = sigma * grad(c)

# Navier-Stokes equations
L_momentum = (
    rho_c(c) * Dt(v) * v_ * dx
    + rho_c(c) * inner(grad(v) * v, v_) * dx
    + inner(T, grad(v_)) * dx
    - rho_c(c) * inner(g, v_) * dx
    - inner(f_cap, v_) * dx  # capillary force
)

L_continuity = div(v) * p_ * dx

L = L_ch + L_momentum + L_continuity

# Compute Jacobian
J = derivative(L, w)

stepper = TimeStepper(L, scheme, t, dt, w, bcs=bcs, options_prefix="")

# Output file
vtk = VTKFile("results/ch_nse_rising_bubble_vpc.pvd")

v, p, c = w.subfunctions
v.rename("velocity")
p.rename("pressure")
c.rename("concentration")

# Initialize concentration
c.interpolate(c_init)

# Time stepping
T = Constant(t_end)

while float(t) < float(T):
    stepper.advance()

    print(f"{float(t)=:4e}")
    vtk.write(v, p, c, time=float(t))

    # Monitor bubble volume (region where c > 0.5)
    V = assemble(conditional(c > 0.5, 1.0, 0.0) * dx)
    c_avg = assemble(c * dx)
    print(f"  volume = {V:e}, c_avg = {c_avg:e}")
