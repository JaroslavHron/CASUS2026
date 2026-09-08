from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
dt_val = opts.getReal('dt', 0.1)  # time step
t_end  = opts.getReal('t_end', 20.0)  # end time

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler

scheme = BackwardEuler()

dt = Constant(dt_val)
t = Constant(0.0)

# Create mesh and define function spaces
n = 20
mesh = RectangleMesh(n, 2*n, 1.0, 2.0)

# inf-sup stable element - Taylor - Hood
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = VectorElement("CG", mesh.ufl_cell(), 2)
El = FiniteElement("CG", mesh.ufl_cell(), 1)

Evpl = MixedElement([Ev, Ep, El])
W = FunctionSpace(mesh, Evpl)

# Define test functions
v_, p_, l_ = TestFunctions(W)

# Define functions
w = Function(W)  # current solution

# Split mixed functions
v, p, l = split(w)

# Boundary conditions
bcv = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), [1, 2, 3, 4])
bcs = [bcv]

# Physical parameters
g = Constant([0.0, -1.0])

# Fluid properties (phase-dependent)
rho1 = 1e2
rho2 = 1e3
mu1 = 1e1
mu2 = 1e0

# Surface tension
st = Constant(1.0)

# Mesh-dependent parameter for smooth sign function
eps = CellDiameter(mesh)

# Spatial coordinates for initial condition
x, y = SpatialCoordinate(mesh)

# Initial distance function (level-set)
center = [0.5, 0.5]
radius = 0.25

bubble1 = sqrt((x - 0.5)**2 + (y - 1.5)**2) - 0.25
bubble2 = sqrt((x - 0.3)**2 + (y - 1.3)**2) - 0.20
base = y - 0.5


def min_func(a, b):
    return conditional(a < b, a, b)


dist = min_func(base, min_func(bubble1, bubble2))


def Sign(q):
    """Smooth sign function avoiding discontinuities"""
    return q / sqrt(q * q + eps * eps)


def rho(l):
    """Density field depending on level-set"""
    return rho1 * 0.5 * (1.0 + Sign(l)) + rho2 * 0.5 * (1.0 - Sign(l))


def nu(l):
    """Viscosity field depending on level-set"""
    return mu1 * 0.5 * (1.0 + Sign(l)) + mu2 * 0.5 * (1.0 - Sign(l))


# Identity and normal
I = Identity(mesh.topological_dimension)
n_facet = FacetNormal(mesh)

# Weak statement of the equations
# Level-set transport: Dt(l) + grad(l)*v = 0
# Momentum: rho*Dt(v) + rho*(grad(v)*v) - div(T) + grad(p) = rho*g + surface tension
# Continuity: div(v) = 0

# Stress tensor
T_stress = -p * I + nu(l) * (grad(v) + grad(v).T)

# Level-set equation
L_levelset = Dt(l) * l_ * dx + inner(grad(l) * v, l_) * dx

# Momentum equation (including Boussinesq buoyancy force for density variations)
L_momentum = (
    rho(l) * Dt(v) * v_ * dx
    + rho(l) * inner(grad(v) * v, v_) * dx
    + inner(T_stress, grad(v_)) * dx
    - rho(l) * inner(g, v_) * dx
)

# Surface tension force
# F_surf = sigma * kappa * n * delta(phi)
# Simplified: sigma * |grad(phi)| on interface, formulated as:
# F_surf = sigma * sqrt(grad(l)*grad(l)) * (I - n_interface * n_interface) * grad(v_)
grad_l_mag = sqrt(inner(grad(l), grad(l)))
n_interface = grad(l) / (grad_l_mag + eps)
L_surfacetension = st * grad_l_mag * inner(I - outer(n_interface, n_interface), grad(v_)) * dx

# Continuity
L_continuity = div(v) * p_ * dx

L = L_levelset + L_momentum + L_surfacetension + L_continuity

# Compute Jacobian
J = derivative(L, w)

stepper = TimeStepper(L, scheme, t, dt, w, bcs=bcs, options_prefix="")

# Output file
vtk = VTKFile("results/rising_bubble_vpl.pvd")

v, p, l = w.subfunctions
v.rename("velocity")
p.rename("pressure")
l.rename("levelset")

# Initialize level-set
l.interpolate(dist)

# Time stepping
T = Constant(t_end)

while float(t) < float(T):
    stepper.advance()

    print(f"{float(t)=:4e}")
    vtk.write(v, p, l, time=float(t))

    # Monitor bubble volume
    V = assemble(conditional(l < 0.0, 1.0, 0.0) * dx)
    print(f"  volume = {V:e}")
