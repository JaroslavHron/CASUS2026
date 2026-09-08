from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
dt_val = opts.getReal('dt', 0.1)  # time step
t_end  = opts.getReal('t_end', 30.0)  # end time
Ra     = opts.getReal('Ra', 1e4)  # Rayleigh number
Pr     = opts.getReal('Pr', 0.71)  # Prandtl number

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler

scheme = BackwardEuler()

dt = Constant(dt_val)
t = Constant(0.0)

# Create mesh and define function spaces
n = 20
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
bce_top = DirichletBC(W.sub(2), 0, [4])
bce_down = DirichletBC(W.sub(2), 1, [3])

bcs = [bcv_wall, bce_top, bce_down]

# Physical parameters
g = Constant([0.0, -1.0])  # nondimensional gravity vector

# Strain rate and stress tensor
D = sym(grad(v))
T = 2 * Constant(Pr / Ra) * D
K = Constant(1.0 / np.sqrt(Ra * Pr))

# Weak statement of the equations
# Momentum equation: Dt(v) + (grad(v)*v) - div(T) + grad(p) = e*g
# Energy equation: Dt(e) + grad(e)*v - div(K*grad(e)) + T:D = 0
# Continuity: div(v) = 0

L_momentum = (
    Dt(v) * v_ * dx
    + inner(grad(v) * v, v_) * dx
    + inner(T, grad(v_)) * dx
    + e * inner(g, v_) * dx
)

L_energy = (
    Dt(e) * e_ * dx
    + inner(v, grad(e)) * e_ * dx
    + inner(K * grad(e), grad(e_)) * dx
    - inner(T, D) * e_ * dx
)

L_continuity = div(v) * p_ * dx

L = L_momentum + L_energy + L_continuity

# Compute Jacobian
J = derivative(L, w)

stepper = TimeStepper(L, scheme, t, dt, w, bcs=bcs, options_prefix="")

# Output file
vtk = VTKFile("results/nse_cavity_vpe.pvd")

v, p, e = w.subfunctions
v.rename("v", "velocity")
p.rename("p", "pressure")
e.rename("e", "temperature")

# Initial conditions (zero everywhere, boundary conditions will be applied)
# This is the default for Firedrake functions

# Time stepping
T = Constant(t_end)

while float(t) < float(T):
    stepper.advance()

    print(f"{float(t)=:4e}")
    vtk.write(v, p, e, time=float(t))
