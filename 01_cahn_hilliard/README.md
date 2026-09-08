# Solving the Cahn-Hilliard Equation with Firedrake

A practical tutorial on solving the **Cahn-Hilliard equation** using Firedrake and the Irksome time-stepping library.

## Overview

This tutorial demonstrates how to solve the Cahn-Hilliard equation, which models phase separation in binary fluid mixtures. The implementation uses a fully implicit time-stepping scheme and includes both a rectangular domain example and guidance for solving the equation on curved surfaces.

## Contents

- [Introduction](#introduction)
- [Mathematical Formulation](#mathematical-formulation)
- [Mixed Formulation](#mixed-formulation)
- [Firedrake Implementation](#firedrake-implementation)
- [Files in This Folder](#files-in-this-folder)
- [Running the Code](#running-the-code)

## Introduction

The **Cahn-Hilliard equation** models the process of **phase separation** (spinodal decomposition) in a binary fluid mixture. It describes how two components naturally separate to form pure domains, driven by the minimization of a free-energy functional. This is a fourth-order nonlinear partial differential equation widely used in materials science and fluid dynamics.

## Mathematical Formulation

The Cahn-Hilliard equation is given by:

$$
\frac{\partial c}{\partial t} = \nabla \cdot \left( M \nabla \mu \right)
$$

where:
- $c$ is the concentration (order parameter)
- $M$ is the mobility (scalar or tensor-valued)
- $\mu$ is the chemical potential

The chemical potential is derived from the Ginzburg-Landau free energy:

$$
\mu = \frac{df}{dc} - \lambda^2 \nabla^2 c
$$

with the double-well potential:

$$
f(c) = \frac{1}{4}(c^2 - 1)^2
$$

The parameter $\lambda$ controls the interfacial thickness and is a key tuning parameter in the simulation.

## Mixed Formulation

To solve the fourth-order PDE using standard $C^0$ continuous finite elements, the equation is split into two coupled second-order equations:

$$
\frac{\partial c}{\partial t} - \nabla \cdot \left( M \nabla \mu \right) = 0
$$

$$
\mu - \frac{df}{dc} + \lambda^2 \nabla^2 c = 0
$$

This mixed formulation is solved using a mixed function space with:
- $c$ (concentration) in a standard Lagrange space
- $\mu$ (chemical potential) in a standard Lagrange space

## Firedrake Implementation

### Key Components

#### 1. Mesh and Function Spaces

```python
mesh = RectangleMesh(80, 80, 1, 1)  # 80x80 structured mesh on unit square

C = FunctionSpace(mesh, "CG", 1)     # Continuous Galerkin, degree 1
M = FunctionSpace(mesh, "CG", 1)     # Same for chemical potential
W = MixedFunctionSpace([C, M])       # Mixed space for (c, mu)
```

#### 2. Initial Conditions

The code initializes concentration using a combination of cosine modes:

```python
k = 2*pi
c_init = 0.5 + 0.05*(cos(k*x)*cos(2*k*y) + cos(3*k*x)*cos(k*y) + cos(2*k*x+0.3))
```

This creates a smooth initial perturbation around $c = 0.5$ to trigger spinodal decomposition.

#### 3. Variational Formulation

The weak form is expressed using Firedrake's Unified Form Language (UFL):

```python
c = variable(c)
f = 0.25*(c**2 - 1)**2           # Free energy density
dfdc = diff(f, c)                 # Automatic differentiation

L0 = Dt(c)*c_*dx + dot(grad(m), grad(c_))*dx
L1 = m*m_*dx - dfdc*m_*dx - lmbda**2*dot(grad(c), grad(m_))*dx
L = L0 + L1
```

Where:
- `Dt(c)` is the automatic time derivative (from Irksome)
- `lmbda` is the interfacial parameter $\lambda$

#### 4. Time Stepping with Irksome

The tutorial uses the **Irksome** library for robust time integration:

```python
from irksome import Dt, TimeStepper, BackwardEuler

scheme = BackwardEuler()  # Fully implicit time-stepping
stepper = TimeStepper(L, scheme, t, dt, w, bcs=[], options_prefix="")

# Time loop
while float(t) < T:
    stepper.advance()
    c_tot = float(assemble(c*dx))  # Monitor total concentration
    vtk.write(c, m, time=float(t))
```

### Time Integration Details

- **Time-stepping scheme**: Backward Euler (first-order, fully implicit)
- **Jacobian computation**: Automatic via `derivative(L, w)`
- **Nonlinear solver**: PETSc SNES (Newton-Krylov method)
- **Solver parameters**: Customizable via PETSc options

## Files in This Folder

### `01_cahn_hilliard.py` (Main solver)

Solves the Cahn-Hilliard equation on a rectangular domain with the following features:

- **Mesh**: $80 \times 80$ structured rectangular mesh on $[0,1]^2$
- **Function spaces**: Mixed CG1-CG1 spaces for $(c, \mu)$
- **Time stepping**: Backward Euler via Irksome
- **Output**: VTK files for ParaView visualization
- **Diagnostics**: Prints total concentration at each time step

**Command-line parameters** (via PETSc options):
- `-lambda`: Interfacial parameter (default: 0.02)
- `-dt`: Time step size (default: 0.01)

Example usage:
```bash
python 01_cahn_hilliard.py -lambda 0.01 -dt 0.005
```

### `sphere_surface.py` (Curved geometry)

Demonstrates mesh generation for solving Cahn-Hilliard on curved surfaces:

```python
from netgen.occ import Sphere
shape = Sphere(Pnt(0,0,0), 1)
ngmesh = OCCGeometry(shape).GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh, netgen_flags={"degree": 2})
```

This file shows how to:
- Use Netgen/OCC geometry kernel for 3D surface meshes
- Generate a unit sphere surface mesh
- Create a degree-2 curved mesh for accurate geometry representation

**Note**: This is a template for extending the solver to curved domains. To use it with the main solver, adapt the mesh creation step.

## Running the Code

### On ROSI (with prepared environment)

```bash
source /home/tut10/hron/CASUS2026/setup.sh
cd 01_cahn_hilliard
srun -n 1 -u python 01_cahn_hilliard.py -lambda 0.02 -dt 0.01
```

### On Google Colab

Use the setup instructions in the main `colab.txt` file, then:

```python
%cd 01_cahn_hilliard
!python 01_cahn_hilliard.py -lambda 0.02 -dt 0.01
```

### Local Installation

Refer to [Firedrake Installation](https://www.firedrakeproject.org/install.html).

**Additional dependencies**:
- **Irksome**: Time-stepping library `pip install irksome`
- **Netgen**: Mesh generation (optional, for `sphere_surface.py`) `pip install netgen-mesher`

## Key Concepts Learned

- **Automatic differentiation** in UFL for computing chemical potential
- **Mixed function spaces** for coupled systems of PDEs
- **Implicit time stepping** with Irksome for improved stability
- **Mesh generation** for 2D rectangular and 3D curved domains
- **Diagnostic monitoring** during long time integrations
- **Parametric studies** using PETSc command-line options

## Physics Insights

1. **Phase separation dynamics**: Initial perturbations grow and evolve into separated phases
2. **Total mass conservation**: The sum $\int_\Omega c \, dx$ should remain constant (numerically verified)
3. **Interfacial dynamics**: Parameter $\lambda$ controls the thickness of the interface between phases
4. **Time-step selection**: Smaller $\Delta t$ is needed for finer interfaces (larger $\lambda$ requires smaller $dt$)

## References

- [Firedrake Project](https://www.firedrakeproject.org/)
- [Irksome Documentation](https://irksome.readthedocs.io/)
- [PETSc](https://petsc.org/)
- [Netgen Mesher](https://github.com/NGSolve/netgen)
- Cahn-Hilliard equation: Classic work on phase separation dynamics
