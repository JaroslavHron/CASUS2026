from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
dt_val = opts.getReal('dt', 0.1)  # time step
t_end  = opts.getReal('t_end', 50.0)  # end time
Re     = opts.getReal('Ra', 1e2)  # Raynolds number

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler, RadauIIA, BDF, ContinuousPetrovGalerkinScheme

#scheme = BackwardEuler()
#scheme = ContinuousPetrovGalerkinScheme(2)
scheme = RadauIIA(2)
#scheme = BDF(2)

print(f"{scheme.__dict__=}")

dt = Constant(dt_val)
t = Constant(0.0)

# Create mesh and define function spaces
n = 40
mesh = UnitSquareMesh(n, n)

k=1
#V = VectorFunctionSpace(mesh, "CG", k+1)
#P = FunctionSpace(mesh, "CG", k)
V = VectorFunctionSpace(mesh, "CG", k+1, variant="alfeld")
P = FunctionSpace(mesh, "DG", k, variant="alfeld")

W = MixedFunctionSpace([V,P])

# Define test functions
v_, p_ = TestFunctions(W)

# Define functions
w = Function(W)  # current solution

# Split mixed functions
v, p = split(w)

# Boundary conditions
bcv_wall = DirichletBC(W.sub(0), as_vector([0, 0]), [1, 2, 3])
bcv_top = DirichletBC(W.sub(0), as_vector([1.0,0.0]), [4])

bcs = [bcv_wall, bcv_top]

# Spatial coordinates for initial condition
x, y = SpatialCoordinate(mesh)

# Strain rate and stress tensor
I = Identity(mesh.topological_dimension)
D = sym(grad(v))
T = -p*I + 2 * Constant( 1.0/Re ) * D

print(f"Re={Re}")

L1 = inner(dot(grad(v), v), v_) * dx + inner(T, grad(v_)) * dx
L2 = div(v) * p_ * dx
F = inner(Dt(v), v_) * dx + L1 + L2

nullsp = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])
stepper = TimeStepper(F, scheme, t, dt, w, bcs=bcs, options_prefix="", nullspace=nullsp)

# Output file
vtk = VTKFile("results/ns_cavity_vp.pvd")

v, p = w.subfunctions
v.rename("v", "velocity")
p.rename("p", "pressure")

# Time stepping
T = Constant(t_end)

print(f"{float(t)=:4e}")
vtk.write(v, p, time=float(t))

v0 = Function(v.function_space())
v0.assign(v)
              

while float(t) < float(T):
    stepper.advance()
    t.assign(t+dt)
    print(stepper.solver_stats())

    dvdt = assemble(L1)
    for bc in bcs: bc.zero(dvdt)
    dvdt = dvdt.riesz_representation()

    ddvdt = (v-v0)/dt
    print(f"{float(t)=:4e} {norm(dvdt)=}  {norm(ddvdt)=}")
    v0.assign(v)
    vtk.write(v, p, time=float(t))
