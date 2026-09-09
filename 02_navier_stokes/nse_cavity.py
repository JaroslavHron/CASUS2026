from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
dt_val = opts.getReal('dt', 0.1)  # time step
t_end  = opts.getReal('t_end', 50.0)  # end time
Ra     = opts.getReal('Ra', 1e5)  # Rayleigh number
Pr     = opts.getReal('Pr', 0.71)  # Prandtl number

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler, RadauIIA, BDF

#scheme = BackwardEuler()
scheme = RadauIIA(2)
#scheme = BDF(2)

print(f"{scheme.__dict__=}")

dt = Constant(dt_val)
t = Constant(0.0)

# Create mesh and define function spaces
n = 40
mesh = UnitSquareMesh(n, n)

# inf-sup stable element - Taylor - Hood
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = VectorElement("CG", mesh.ufl_cell(), 2)
Ee = FiniteElement("CG", mesh.ufl_cell(), 2)

Evpe = MixedElement([Ev, Ep, Ee])
W = FunctionSpace(mesh, Evpe)

# Define test functions
v_, p_, e_ = TestFunctions(W)

# Define functions
w = Function(W)  # current solution

# Split mixed functions
v, p, e = split(w)

# Boundary conditions
bcv_wall = DirichletBC(W.sub(0), as_vector([0, 0]), [1, 2, 3, 4])
bce_top = DirichletBC(W.sub(2), 0.0, [4])
bce_down = DirichletBC(W.sub(2), 1.0, [3])
bce_left = DirichletBC(W.sub(2), 1.0, [1])
bce_right = DirichletBC(W.sub(2), 0.0, [2])

#bcs = [bcv_wall, bce_top, bce_down]
bcs = [bcv_wall, bce_left, bce_right]

# Spatial coordinates for initial condition
x, y = SpatialCoordinate(mesh)

# Physical parameters
g = Constant([0.0, -1.0])  # nondimensional gravity vector

# Strain rate and stress tensor
I = Identity(mesh.topological_dimension)
D = sym(grad(v))
T = -p*I + 2 * Constant( np.sqrt(Pr / Ra)) * D
K = Constant(1.0 / np.sqrt(Ra * Pr))

print(f"Re={1/np.sqrt(Pr/Ra)} T_diff={1/np.sqrt(Ra * Pr)}")

# Weak statement of the equations
# Momentum equation: Dt(v) + (grad(v)*v) - div(T) + grad(p) = e*g
# Energy equation: Dt(e) + grad(e)*v - div(K*grad(e)) + T:D = 0
# Continuity: div(v) = 0

L_momentum = (
    inner(Dt(v), v_) * dx
    + inner(dot(grad(v), v), v_) * dx
    + inner(T, grad(v_)) * dx
    + inner(e*g, v_) * dx
)

L_energy = (
    inner(Dt(e), e_) * dx
    + inner(dot(grad(e),v), e_) * dx
    + inner(K * grad(e), grad(e_)) * dx
    #- inner( inner(T, D), e_) * dx
)

L_continuity = div(v) * p_ * dx

L = L_momentum + L_energy + L_continuity

# Compute Jacobian
J = derivative(L, w)

nullsp = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm), W.sub(2)])

stepper = TimeStepper(L, scheme, t, dt, w, bcs=bcs, options_prefix="", nullspace=nullsp)

# Output file
vtk = VTKFile("results/nse_cavity_vpe.pvd")

v, p, e = w.subfunctions
v.rename("v", "velocity")
p.rename("p", "pressure")
e.rename("e", "temperature")

# Initial conditions (zero everywhere, boundary conditions will be applied)
# Initialize temperature
#e.interpolate(1-y)
e.interpolate(1-x)

# This is the default for Firedrake functions

# Time stepping
T = Constant(t_end)

print(f"{float(t)=:4e}")
vtk.write(v, p, e, time=float(t))

while float(t) < float(T):
    stepper.advance()
    t.assign(t+dt)
    
    print(f"{float(t)=:4e}")
    vtk.write(v, p, e, time=float(t))
