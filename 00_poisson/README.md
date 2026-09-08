# Poisson Equation Tutorial

A simple introduction to solving the **Poisson equation** using Firedrake, a Python-based finite element package.

## Overview

This tutorial demonstrates the fundamental concepts of solving a boundary value problem using the finite element method (FEM). The Poisson equation is one of the most basic partial differential equations and serves as an ideal starting point for learning Firedrake.

## Contents

- [Problem Description](#problem-description)
- [Mathematical Formulation](#mathematical-formulation)
- [Firedrake Implementation](#firedrake-implementation)
- [Running the Code](#running-the-code)

## Problem Description

The tutorial solves the Poisson equation on a unit square domain with zero Dirichlet boundary conditions:

$$
-\Delta u = f \quad \text{in } \Omega \\
u = 0 \quad \text{on } \partial\Omega
$$

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

$$\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f \, v \, dx \quad \forall v \in V$$

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

## Running the Code

### On ROSI (with prepared environment)

```bash
source /home/tut10/hron/CASUS2026/setup.sh
srun -n 1 -u python 00_poisson/00_simple_poisson.py
```

### On Google Colab

Use the setup instructions in the main `colab.txt` file.

### Local Installation

Refer to [Firedrake Installation](https://www.firedrakeproject.org/install.html) for installation instructions on your system.

## Files

- `00_simple_poisson.py`: Main solver script demonstrating the solution of the Poisson equation

## Key Concepts Introduced

- Mesh generation in Firedrake
- Function spaces and test/trial functions
- Variational formulation and the Unified Form Language (UFL)
- Boundary conditions
- Solving linear variational problems
- Error computation
- Visualization with ParaView

## References

- [Firedrake Project](https://www.firedrakeproject.org/)
- [PETSc](https://petsc.org/) - Underlying linear solver library
