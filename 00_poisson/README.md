## Problem Description

The tutorial solves the Poisson equation on a unit square domain with zero Dirichlet boundary conditions:

```math
\begin{aligned}
-\Delta u &= f \quad \text{in } \Omega \\
u &= 0 \quad \text{on } \partial\Omega \\
\end{aligned}
```

where:
- $\Omega = [0,1] \times [0,1]$ is the unit square domain
- $f$ is the source term
- $u$ is the unknown solution

## Mathematical Formulation

### Exact Solution

The tutorial uses a manufactured solution:

$$u_{\text{ex}}(x,y) = x(1-x)y(1-y)$$

This allows easy verification of the numerical solution by computing the error.

### Weak Formulation

The variational (weak) form of the Poisson equation is:

$$\int_\Omega \nabla u \cdot \nabla v \  dx = \int_\Omega f \cdot v \  dx \quad \forall v \in V$$

where $V$ is the space of test functions satisfying the homogeneous Dirichlet boundary conditions.

## Firedrake Implementation

### Key Steps

1. **Create a mesh**: Generate a structured mesh of the unit square
2. **Define function spaces**: Set up Lagrange finite element spaces
3. **Formulate the weak form**: Define the variational problem using Firedrake's Unified Form Language (UFL)
4. **Apply boundary conditions**: Enforce zero Dirichlet boundary conditions
5. **Solve**: Use Firedrake's linear solver to find the numerical solution
6. **Post-process**: Compute errors and visualize results

### Output

The program:
- Saves the solution in VTK format for visualization in ParaView
- Computes the $L^2$ and $H^1$ norms of the error
- Generates a contour plot of the solution as a PDF
