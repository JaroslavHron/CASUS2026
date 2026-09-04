from firedrake import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "Lagrange", 1)

x,y = SpatialCoordinate(mesh)

uex = x*(1-x)*y*(1-y)
f = -div(grad(uex))

bc = DirichletBC(V, 0.0, "on_boundary")

# Prepare variational formulation
u = Function(V, name="uh")
v = TestFunction(V)

F = inner(grad(u),grad(v))*dx - f*v*dx

# Init function and solve variational problem
solve(F == 0, u, bc)

# Save for paraview
VTKFile(f"results/u.pvd").write(u)

# Compute the errors
err_L2= norm(u-uex, "L2")
#err_L2= sqrt( assemble( ((u-uex)**2)*dx ) )
err_H1= norm(u-uex, "H1")
    
print(f"Final error L2        {err_L2=:16.8e}")
print(f"Final error H1        {err_H1=:16.8e}")


from firedrake import PETSc
rank = PETSc.COMM_WORLD.rank

# Plot the solution
import matplotlib.pyplot as plt
fig, axes = plt.subplots()
p = tricontour(u,axes=axes)
fig.colorbar(p);
plt.savefig(f'fig_{rank}_u.pdf', bbox_inches='tight')
