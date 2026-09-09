from firedrake import *
from firedrake import PETSc
import numpy as np

print = PETSc.Sys.Print
opts = PETSc.Options()

# Model parameters
lmbda  = opts.getReal('lambda', 2.0e-02)  # surface parameter
dt     = opts.getReal('dt', 1.0e-02)  # time step

# Time stepping library irksome
from irksome import Dt, TimeStepper, BackwardEuler, BDF, ContinuousPetrovGalerkinScheme
scheme = BackwardEuler()

dt = Constant(dt)
t = Constant(0.0)

# Create mesh and define function spaces
mesh = RectangleMesh(80, 80, 1, 1) #, quadrilateral=True)

C = FunctionSpace(mesh, "CG", 1)
M = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([C,M])

# Define trial and test functions
c_, m_  = TestFunctions(W)

# Define functions
w   = Function(W)  # current solution

# Split mixed functions
c, m  = split(w)

# Compute the chemical potential df/dc
c=variable(c)
f = 0.25*(c**2-1)**2
dfdc = diff(f, c)

# Weak statement of the equations
L0 = Dt(c)*c_*dx + dot(grad(m), grad(c_))*dx
L1 = m*m_*dx - dfdc*m_*dx - lmbda**2*dot(grad(c), grad(m_))*dx
L = L0 + L1

# Compute directional derivative about u in the direction of du (Jacobian)
J = derivative(L, w)

bcs=[]

stepper = TimeStepper(L, scheme, t, dt, w, bcs=bcs, options_prefix="")

# Output file
vtk = VTKFile("results/ch.pvd")

c, m  = w.subfunctions
c.rename("c")
m.rename("m")

# Create intial conditions and interpolate
x, y = SpatialCoordinate(mesh)

#pcg = PCG64(seed=42)
#rg = Generator(pcg)
# 3. Sample a random function directly into your space using a distribution
# This populates the function's degrees of freedom (DoFs) automatically
#noise = rg.normal(M, 0.0, 1.0)
#c_init = 0.63+0.2*noise #sin(x)*sin(y)
k=2*np.pi
c_init = 0.5 + 0.05*(cos(k*x)*cos(2*k*y)+cos(3*k*x)*cos(k*y)+cos(2*k*x+0.3))

c.interpolate(c_init)


# Step in time
T = 200*float(dt)
while (float(t) < T):

    stepper.advance()
    
    c_tot = float(assemble(c*dx))
    print(f"{float(t)=:4e} {c_tot=:4e}")
    t.assign(t + dt)
    vtk.write(c, m, time=float(t))
    

                                                                                                                                                                                
