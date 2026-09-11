from firedrake import *
import numpy as np

# 1. Mesh Definition (Parameter h)
N = 64
mesh = UnitSquareMesh(N, N)
h = 1.0 / N

# 2. Mixed Function Spaces (Taylor-Hood P2-P1 for fluid stability)
V = FunctionSpace(mesh, "Lagrange", 1)
W = VectorFunctionSpace(mesh, "Lagrange", 2)
P = FunctionSpace(mesh, "Lagrange", 1)

# Combined Mixed Space: [c1, c2, mu1, mu2, u, p]
MS = MixedFunctionSpace([V, V, V, V, W, P])

# Trial and Test functions
sol = Function(MS)
c1, c2, mu1, mu2, u, p = split(sol)
v1, v2, q1, q2, phi, psi = TestFunctions(MS)

# Previous timestep solution
sol_n = Function(MS)
c1_n, c2_n, _, _, _, _ = split(sol_n)

# 3. Nondimensional Parameters (from Table 2 of the paper)
dt = Constant(2e-4)
eps = Constant(2.5 * h)  # Interfacial thickness proportional to mesh h
Ca = Constant(1e-3)     # Capillary number
Bo = Constant(1e-3)     # Bond number (controls buoyancy gravity strength)
  
# Surface Tensions (scaled such that \sigma_12 + \sigma_13 + \sigma_23 = 1)
sigma12 = Constant(0.0636)
sigma13 = Constant(0.4983)
sigma23 = Constant(0.4381)

# Density Perturbations (from Table 2)
d1 = Constant(0.05)
d2 = Constant(-0.01)
d3 = Constant(0.6)

# Mobility Tensor parameters (m_ii = 2, m_ij = -1)
m11, m12 = Constant(2.0), Constant(-1.0)
m21, m22 = Constant(-1.0), Constant(2.0)

# 4. Physical Expressions (Bulk Potential Derivatives, Density, & Surface Energy)
# Dependent third phase concentration
c3 = 1.0 - c1 - c2

# Bulk derivative driving forces (f'_i = 2 * c_i * (1-c_i) * (1-2c_i))
f_prime1 = 2.0 * c1 * (1.0 - c1) * (1.0 - 2.0 * c1)
f_prime2 = 2.0 * c2 * (1.0 - c2) * (1.0 - 2.0 * c2)
f_prime3 = 2.0 * c3 * (1.0 - c3) * (1.0 - 2.0 * c3)

# Perturbation density equation for Boussinesq term: rho = \sum (d_i * c_i)
rho = d1 * c1 + d2 * c2 + d3 * c3
g_vec = Constant((0.0, -1.0))  # Gravitational unit vector acting downwards

# 5. Mixed Weak Formulations
# Cahn-Hilliard Transport Equations
F_c1 = ((c1 - c1_n) / dt) * v1 * dx + dot(u, grad(c1)) * v1 * dx \
       + (m11 * dot(grad(mu1), grad(v1)) + m12 * dot(grad(mu2), grad(v1))) * dx

F_c2 = ((c2 - c2_n) / dt) * v2 * dx + dot(u, grad(c2)) * v2 * dx \
       + (m21 * dot(grad(mu1), grad(v2)) + m22 * dot(grad(mu2), grad(v2))) * dx

# Chemical Potential Definitions (incorporating surface tensions Matrix \sigma_ij)
F_mu1 = mu1 * q1 * dx - f_prime1 * q1 * dx \
        + (eps**2) * (sigma12 * dot(grad(c2), grad(q1)) + sigma13 * dot(grad(c3), grad(q1))) * dx

F_mu2 = mu2 * q2 * dx - f_prime2 * q2 * dx \
        + (eps**2) * (sigma12 * dot(grad(c1), grad(q2)) + sigma23 * dot(grad(c3), grad(q2))) * dx

# Stokes Flow: Includes Capillary Surface Force and the new Buoyancy Force
mu3 = f_prime3  
capillary_force = (3.0 / (np.sqrt(2) * eps)) * (mu1 * grad(c1) + mu2 * grad(c2) + mu3 * grad(c3))
buoyancy_force = (Bo / eps) * rho * g_vec  # Rescaled gravitational force term

F_stokes = Ca * inner(grad(u), grad(phi)) * dx - p * div(phi) * dx \
           - dot(capillary_force, phi) * dx - dot(buoyancy_force, phi) * dx
F_div = div(u) * psi * dx

# Total System Residual
F = F_c1 + F_c2 + F_mu1 + F_mu2 + F_stokes + F_div

# Boundary Conditions (No-slip walls for velocity component)
bcs = [DirichletBC(MS.sub(4), Constant((0.0, 0.0)), "on_boundary")]

# 6. Initialize Configuration (Mixed initial state with small noise)
sol_init = Function(MS)
c1_init, c2_init = sol_init.sub(0), sol_init.sub(1)

np.random.seed(42)
c1_init.dat.data[:] = 0.5 + 0.02 * (2.0 * np.random.rand(len(c1_init.dat.data)) - 1.0)
c2_init.dat.data[:] = 0.5 + 0.02 * (2.0 * np.random.rand(len(c2_init.dat.data)) - 1.0)

# Synchronize current and previous time steps
sol.assign(sol_init)
sol_n.assign(sol_init)

# 7. Nonlinear Solver Configuration
problem = NonlinearVariationalProblem(F, sol, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters={
    "snes_type": "newtonls",
    "snes_rtol": 1e-6,
    "snes_max_it": 40,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
})

# 8. Time-Stepping Solver Loop
t = 0.0
T_max = 0.05

while t < T_max:
    t += float(dt)
    solver.solve()
    sol_n.assign(sol)
    print(f"Time Step: t = {t:.4f}")
