# Solving the Cahn-Hilliard Equation with Firedrake

A tutorial on solving the **Cahn-Hilliard equation** using the finite element package [Firedrake](https://www.firedrakeproject.org/).

> Converted to Markdown from the supplied presentation *Solving the Cahn-Hilliard Equation: A Tutorial using Firedrake* (September 3, 2026).

## Contents

- [Introduction](#introduction)
- [Mathematical formulation](#mathematical-formulation)
- [Mixed formulation](#mixed-formulation)
- [Time discretization](#time-discretization-theta-method)
- [Firedrake implementation](#firedrake-implementation)
  - [Mesh and function spaces](#1-mesh-and-function-spaces)
  - [Initial conditions](#2-initial-conditions)
  - [Weak form](#3-weak-form)
  - [Solving the nonlinear system](#4-solving-the-nonlinear-system)
- [Complete example](#complete-example)
- [Summary](#summary)

## Introduction

The **Cahn-Hilliard equation** models the process of phase separation, or **spinodal decomposition**, in a binary fluid mixture.

It:

- describes how two components separate to form pure domains,
- is driven by minimization of a free-energy functional,
- is a fourth-order nonlinear partial differential equation.

## Mathematical formulation

The standard Cahn-Hilliard equation is

$$
\frac{\partial c}{\partial t}
= \nabla \cdot \left(M\nabla\mu\right),
$$

where

- $c$ is the concentration, ranging approximately from $-1$ to $1$,
- $M$ is the mobility,
- $\mu$ is the chemical potential.

The chemical potential is derived from the Ginzburg-Landau free energy:

$$
\mu = \frac{df}{dc} - \lambda \nabla^2 c,
$$

with the double-well potential

$$
f(c) = \frac14(c^2-1)^2.
$$

The parameter $\lambda$ is related to the interfacial thickness.

## Mixed formulation

To solve the fourth-order PDE in Firedrake using standard $C^0$ continuous finite elements, split it into two coupled second-order equations:

$$
\frac{\partial c}{\partial t}
- \nabla\cdot(M\nabla\mu) = 0,
$$

$$
\mu - (c^3-c) + \lambda\nabla^2 c = 0.
$$

Seek

$$
c,\mu \in V.
$$

Multiplying by test functions $v,q\in V$ and integrating by parts gives the weak formulation.

## Time discretization ($\theta$-method)

Use a semi-implicit or fully implicit time-stepping scheme. Let $\Delta t$ denote the time step.

The first equation becomes

$$
\int_\Omega
\left(
\frac{c^{n+1}-c^n}{\Delta t}v
+ M\nabla\mu^{n+\theta}\cdot\nabla v
\right)\,dx = 0,
$$

and the second equation is

$$
\int_\Omega
\left(
\mu^{n+1}q
- \left((c^{n+1})^3-c^{n+1}\right)q
- \lambda\nabla c^{n+1}\cdot\nabla q
\right)\,dx = 0.
$$

Here

$$
\mu^{n+\theta}
= (1-\theta)\mu^n + \theta\mu^{n+1}.
$$

Special cases:

- $\theta=0.5$: Crank-Nicolson,
- $\theta=1$: Backward Euler.

## Firedrake implementation

### 1. Mesh and function spaces

Create a two-dimensional unit-square mesh and a mixed function space for $c$ and $\mu$.

```python
from firedrake import *

# Create a 2D mesh
mesh = UnitSquareMesh(96, 96)

# Define the Function Space (Mixed for c and mu)
V = FunctionSpace(mesh, "CG", 1)
W = V * V

# Define trial and test functions
v, q = TestFunctions(W)
```

### 2. Initial conditions

Initialize the concentration using small random perturbations around a zero mean to trigger spinodal decomposition.

```python
import numpy as np

# Define Functions for current and previous time steps
u = Function(W)
u0 = Function(W)

# Split into c and mu to manipulate components
c, mu = u.subfunctions
c0, mu0 = u0.subfunctions

# Initial condition: c = 0.0 + random noise
c_init = 0.0 + 0.02 * (2 * np.random.rand(V.dim()) - 1)
c.dat.data[:] = c_init
u0.assign(u)
```

### 3. Weak form

Use the **Unified Form Language (UFL)** to define the residual equations.

```python
dt = Constant(5e-6)
lmbda = Constant(1e-2)
theta = Constant(0.5)

# Re-split for symbolic UFL formulation
c, mu = split(u)
c0, mu0 = split(u0)

mu_mid = (1.0 - theta) * mu0 + theta * mu

# Weak form equations
F0 = c * v * dx - c0 * v * dx \
     + dt * dot(grad(mu_mid), grad(v)) * dx

F1 = mu * q * dx \
     - (c**3 - c) * q * dx \
     - lmbda * dot(grad(c), grad(q)) * dx

F = F0 + F1
```

### 4. Solving the nonlinear system

PETSc's **SNES** nonlinear solver is used at each time step.

```python
# Define the nonlinear variational problem
prob = NonlinearVariationalProblem(F, u)

solver = NonlinearVariationalSolver(
    prob,
    solver_parameters={
        'snes_type': 'newtonls',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    },
)

# Time stepping loop
t = 0.0
T = 0.001

while t < T:
    solver.solve()
    u0.assign(u)
    t += float(dt)
```

## Complete example

The code fragments above can be combined into the following compact Firedrake program:

```python
from firedrake import *
import numpy as np

# Mesh and mixed function space
mesh = UnitSquareMesh(96, 96)
V = FunctionSpace(mesh, "CG", 1)
W = V * V

v, q = TestFunctions(W)

# Current and previous solutions
u = Function(W)
u0 = Function(W)

# Initial concentration
c, mu = u.subfunctions
c0, mu0 = u0.subfunctions

c_init = 0.0 + 0.02 * (2 * np.random.rand(V.dim()) - 1)
c.dat.data[:] = c_init
u0.assign(u)

# Parameters
dt = Constant(5e-6)
lmbda = Constant(1e-2)
theta = Constant(0.5)

# Symbolic split for UFL
c, mu = split(u)
c0, mu0 = split(u0)

mu_mid = (1.0 - theta) * mu0 + theta * mu

# Residual
F0 = c * v * dx - c0 * v * dx \
     + dt * dot(grad(mu_mid), grad(v)) * dx

F1 = mu * q * dx \
     - (c**3 - c) * q * dx \
     - lmbda * dot(grad(c), grad(q)) * dx

F = F0 + F1

# Nonlinear solver
prob = NonlinearVariationalProblem(F, u)
solver = NonlinearVariationalSolver(
    prob,
    solver_parameters={
        'snes_type': 'newtonls',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    },
)

# Time integration
t = 0.0
T = 0.001

while t < T:
    solver.solve()
    u0.assign(u)
    t += float(dt)
```

## Summary

- Firedrake allows a concise expression of the Cahn-Hilliard equation using UFL.
- Splitting the fourth-order equation into a mixed system avoids the need for $C^1$ continuous elements or discontinuous Galerkin methods.
- PETSc handles the resulting coupled nonlinear algebraic system.
