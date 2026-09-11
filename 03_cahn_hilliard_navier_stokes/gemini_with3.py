from firedrake import *
import numpy as np

# 1. Mesh Definition
N = 64
mesh = UnitSquareMesh(N, N)
h = 1.0 / N

# 2. Mixed Function Spaces
V = FunctionSpace(mesh, "Lagrange", 1)  # Space for scalar components
W = VectorFunctionSpace(mesh, "Lagrange", 2)  # Taylor-Hood Velocity
P = FunctionSpace(mesh, "Lagrange", 1)  # Pressure

# Combined Mixed Space: [c1, c2, c3, mu1, mu2, mu3, gamma, u, p]
MS = MixedFunctionSpace([V, V, V, V, V, V, V, W, P])

# Trial and Test functions
sol = Function(MS)
c1, c2, c3, mu1, mu2, mu3, gamma, u, p = split(sol)
v1, v2, v3, q1, q2, q3, eta, phi, psi = TestFunctions(MS)

# Previous timestep solution
sol_n = Function(MS)
c1_n, c2_n, c3_n, _, _, _, _, _, _ = split(sol_n)

# 3. Nondimensional Parameters
dt = Constant(2e-4)
eps = Constant(2.5 * h)
Ca = Constant(1e-3)
Bo = Constant(1e-3)
  
# Surface Tensions Matrix elements (from Table 2)
sigma12 = Constant(0.0636)
sigma13 = Constant(0.4983)
sigma23 = Constant(0.4381)

# Density Perturbations
d1 = Constant(0.05)
d2 = Constant(-0.01)
d3 = Constant(0.6)

# Standard Symmetric Mobility Tensor for ternary components (m_ii = 2, m_ij = -1)
# J_i = -\sum m_ij \nabla \mu_j
def mobility_flux(v_test, mu1, mu2, mu3):
    flux = (2.0 * dot(grad(mu1), grad(v_test)) - dot(grad(mu2), grad(v_test)) - dot(grad(mu3), grad(v_test)))
    return flux

# 4. Physical Expressions
# Bulk derivative driving forces (f'_i = 2 * c_i * (1-c_i) * (1-2c_i))
f_prime1 = 2.0 * c1 * (1.0 - c1) * (1.0 - 2.0 * c1)
f_prime2 = 2.0 * c2 * (1.0 - c2) * (1.0 - 2.0 * c2)
f_prime3 = 2.0 * c3 * (1.0 - c3) * (1.0 - 2.0 * c3)

# Boussinesq parameters
rho = d1 * c1 + d2 * c2 + d3 * c3
g_vec = Constant((0.0, -1.0))

# 5. Mixed Weak Formulations
# Cahn-Hilliard Transport Equations for all three species
F_c1 = ((c1 - c1_n) / dt) * v1 * dx + dot(u, grad(c1)) * v1 * dx + mobility_flux(v1, mu1, mu2, mu3) * dx
F_c2 = ((c2 - c2_n) / dt) * v2 * dx + dot(u, grad(c2)) * v2 * dx + mobility_flux(v2, mu2, mu1, mu3) * dx
F_c3 = ((c3 - c3_n) / dt) * v3 * dx + dot(u, grad(c3)) * v3 * dx + mobility_flux(v3, mu3, mu1, mu2) * dx

# Chemical Potential Definitions with gamma (Lagrange multiplier) acting as a projection constraint
# Form layout: \sum \sigma_ij * \Delta c_j
F_mu1 = mu1 * q1 * dx - (f_prime1 + gamma) * q1 * dx \
        + (eps**2) * (sigma12 * dot(grad(c2), grad(q1)) + sigma13 * dot(grad(c3), grad(q1))) * dx

F_mu2 = mu2 * q2 * dx - (f_prime2 + gamma) * q2 * dx \
        + (eps**2) * (sigma12 * dot(grad(c1), grad(q2)) + sigma23 * dot(grad(c3), grad(q2))) * dx

F_mu3 = mu3 * q3 * dx - (f_prime3 + gamma) * q3 * dx \
        + (eps**2) * (sigma13 * dot(grad(c1), grad(q3)) + sigma23 * dot(grad(c2), grad(q3))) * dx

# Strict Linear algebraic constraint equation: \sum c_i = 1
F_constraint = (c1 + c2 + c3 - 1.0) * eta * dx

# Stokes Flow equations with Capillary Force and Buoyancy Force
capillary_force = (3.0 / (np.sqrt(2) * eps)) * (mu1 * grad(c1) + mu2 * grad(c2) + mu3 * grad(c3))
buoyancy_force = (Bo / eps) * rho * g_vec

F_stokes = Ca * inner(grad(u), grad(phi)) * dx - p * div(phi) * dx \
           - dot(capillary_force, phi) * dx - dot(buoyancy_force, phi) * dx
F_div = div(u) * psi * dx

# Combined global residual system
F = F_c1 + F_c2 + F_c3 + F_mu1 + F_mu2 + F_mu3 + F_constraint + F_stokes + F_div

# Boundary Conditions (No-slip walls for velocity component at index 7)
bcs = [DirichletBC(MS.sub(7), Constant((0.0, 0.0)), "on_boundary")]

# 6. Initialize Concentrations (Symmetric mixture with noise, summing to exactly 1.0)
sol_init = Function(MS)
c1_init, c2_init, c3_init = sol_init.sub(0), sol_init.sub(1), sol_init.sub(2)

# Generate noise profiles
np.random.seed(42)
noise1 = 0.02 * (2.0 * np.random.rand(len(c1_init.dat.data)) - 1.0)
noise2 = 0.02 * (2.0 * np.random.rand(len(c2_init.dat.data)) - 1.0)

# Set base mixed state: c1=0.33, c2=0.33, c3=0.34
c1_init.dat.data[:] = 0.33 + noise1
c2_init.dat.data[:] = 0.33 + noise2
c3_init.dat.data[:] = 1.0 - c1_init.dat.data[:] - c2_init.dat.data[:]

# Synchronize steps
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

# 8. Solver Loop
t = 0.0
T_max = 0.05

while t < T_max:
    t += float(dt)
    solver.solve()
    sol_n.assign(sol)
    print(f"Time Step: t = {t:.4f}")

    
