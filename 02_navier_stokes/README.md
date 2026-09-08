# Navier-Stokes Equations with Firedrake

A comprehensive tutorial on solving incompressible flow problems using Firedrake, including cavity flow with Boussinesq convection and multiphase flow with level-set methods.

## Overview

This folder contains two advanced examples of solving the Navier-Stokes equations:

1. **`nse_cavity.py`**: Cavity flow with Boussinesq thermal convection (Rayleigh-Bernard problem)
2. **`rising_bubble.py`**: Multiphase flow with surface tension using level-set method

Both examples use **inf-sup stable finite elements** (Taylor-Hood), **implicit time-stepping**, and **PETSc nonlinear solvers**.

## Contents

- [Mathematical Formulation](#mathematical-formulation)
- [Boussinesq Convection](#boussinesq-convection)
- [Multiphase Flow with Level-Set](#multiphase-flow-with-level-set)
- [Firedrake Implementation](#firedrake-implementation)
- [Files in This Folder](#files-in-this-folder)
- [Running the Code](#running-the-code)

## Mathematical Formulation

### Incompressible Navier-Stokes Equations

The fundamental equations solved are the incompressible Navier-Stokes equations:

$$
\frac{\partial \vec{v}}{\partial t} + (\nabla \vec{v}) \vec{v} - \nabla \cdot \mathbf{T} + \nabla p = \vec{f}
$$

$$
\nabla \cdot \vec{v} = 0
$$

where:
- $\vec{v}$ is the velocity field
- $p$ is the pressure
- $\mathbf{T}$ is the stress tensor
- $\vec{f}$ is the body force

## Boussinesq Convection

### Problem Description

The **Boussinesq approximation** couples momentum and energy equations for thermal convection. This reproduces the classic [Rayleigh-Bernard convection problem](https://doi.org/10.1016/j.crme.2008.02.004).

### Governing Equations

The nondimensional system is:

$$
\begin{aligned}
\frac{\partial \vec{v}}{\partial t} + (\nabla \vec{v}) \vec{v} - \nabla \cdot \mathbf{T} + \nabla p &= e \vec{g} \quad\text{in } \Omega \\
\frac{\partial e}{\partial t} + \nabla e \cdot \vec{v} - \nabla \cdot \left(\frac{1}{\sqrt{\mathrm{Pr}\mathrm{Ra}}} \nabla e\right) + \mathbf{T}:\mathbf{D} &= 0 \quad\text{in } \Omega \\
\nabla \cdot \vec{v} &= 0 \quad\text{in } \Omega
\end{aligned}
$$

where:
- $e$ is the temperature (excess temperature)
- $\mathbf{D} = \frac{1}{2}(\nabla\vec{v} + \nabla\vec{v}^T)$ is the strain rate tensor
- $\mathbf{T} = 2 \mathrm{Pr}/\mathrm{Ra} \cdot \mathbf{D}$ is the viscous stress tensor
- $\mathrm{Ra}$ is the Rayleigh number (controls convection strength)
- $\mathrm{Pr}$ is the Prandtl number (viscosity-to-diffusivity ratio)
- $\vec{g} = [0, -1]$ is the nondimensional gravity vector

### Boundary Conditions

- **Walls** (all four boundaries): No-slip velocity condition ($\vec{v} = 0$)
- **Top wall** (y=1): Cold boundary ($e = 0$)
- **Bottom wall** (y=0): Hot boundary ($e = 1$)

### Cavity Flow Problem

The cavity flow is confined in a unit square with:
- Mesh: $20 \times 20$ unit square
- Rayleigh number: $\mathrm{Ra} = 10^4$
- Prandtl number: $\mathrm{Pr} = 0.71$

This is a benchmark problem demonstrating heat-driven circulation in an enclosed cavity.

## Multiphase Flow with Level-Set

### Problem Description

The **rising bubble** benchmark simulates the motion of bubbles in a surrounding fluid using the **level-set method** for interface tracking. This example reproduces the benchmark from the [FeatFlow database](https://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/bubble/bubble_configurations.html).

### Key Features

- **Level-set representation**: Interface between fluids defined by $\phi = 0$ level-set
- **Two-phase flow**: Different densities and viscosities for each phase
- **Surface tension**: Capillary forces at the interface
- **Reinitialization**: Maintaining $|\nabla \phi| = 1$ during evolution

### Governing Equations

$$
\begin{aligned}
\frac{\partial \phi}{\partial t} + \nabla \phi \cdot \vec{v} &= 0 \quad\text{in } \Omega \\
\rho(\phi) \frac{\partial \vec{v}}{\partial t} + \rho(\phi) (\nabla \vec{v}) \vec{v} - \nabla \cdot \mathbf{T} + \nabla p &= \rho(\phi) \vec{g} + \vec{f}_{\text{surf}} \\
\nabla \cdot \vec{v} &= 0 \quad\text{in } \Omega
\end{aligned}
$$

where:
- $\phi$ is the level-set function (distance function)
- $\rho(\phi)$ is the density field (changes across the interface)
- $\mathbf{T}$ is the viscous stress tensor
- $\vec{f}_{\text{surf}} = \sigma \kappa \vec{n} \delta(\phi)$ is the surface tension force
- $\sigma$ is the surface tension coefficient

### Density and Viscosity

Smooth transitions across the interface using a sign function:

$$
\rho(\phi) = \rho_1 \cdot \frac{1 + \text{Sign}(\phi)}{2} + \rho_2 \cdot \frac{1 - \text{Sign}(\phi)}{2}
$$

where the smooth sign function is:

$$
\text{Sign}(\phi) = \frac{\phi}{\sqrt{\phi^2 + \epsilon^2}}
$$

### Domain and Initial Conditions

- **Domain**: $1.0 \times 2.0$ rectangular region (tall for bubble rise)
- **Mesh**: $20 \times 40$ structured mesh
- **Initial bubbles**: 
  - Bubble 1: Center at (0.5, 1.5), radius 0.25
  - Bubble 2: Center at (0.3, 1.3), radius 0.20
  - Base liquid: $y \geq 0.5$
- **Fluid properties**:
  - Fluid 1 (upper): $\rho_1 = 100$, $\mu_1 = 10$
  - Fluid 2 (lower): $\rho_2 = 1000$, $\mu_2 = 1$

## Firedrake Implementation

### Finite Element Spaces

Both examples use **inf-sup stable** Taylor-Hood elements:

```python
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)      # Pressure: CG1
Ev = VectorElement("CG", mesh.ufl_cell(), 2)     # Velocity: CG2 vector
Ee = FiniteElement("CG", mesh.ufl_cell(), 2)     # Energy/Level-set: CG2 or CG1

Evpe = MixedElement([Ev, Ep, Ee])
W = FunctionSpace(mesh, Evpe)
```

This choice ensures:
- **Pressure-velocity stability** (inf-sup condition)
- **Second-order spatial accuracy** for velocity
- **No spurious pressure modes**

### Time Integration

Both examples use **theta-method** (semi-implicit time-stepping):

$$
\frac{w^{n+1} - w^n}{\Delta t} + \theta F(w^{n+1}) + (1-\theta) F(w^n) = 0
$$

- **`nse_cavity.py`**: $\theta = 1.0$ (Backward Euler, fully implicit)
- **`rising_bubble.py`**: $\theta = 0.5$ (Crank-Nicolson)

### Nonlinear Solver

Both examples use **PETSc SNES** with:

```python
lu = {
    "snes_type": "newtonls",           # Newton line-search
    "snes_max_it": 40,                 # Max iterations
    "snes_rtol": 1e-10,                # Relative tolerance
    "snes_atol": 1e-10,                # Absolute tolerance
    "ksp_type": "preonly",             # No iterative solver
    "pc_type": "lu",                   # Direct LU factorization
    "pc_factor_mat_solver_type": "mumps"  # MUMPS solver
}
```

### Null Space Handling

Since the pressure is determined only up to a constant, a null-space basis is specified:

```python
nullsp = MixedVectorSpaceBasis(W, [
    W.sub(0),                              # Velocity: no null space
    VectorSpaceBasis(constant=True, comm), # Pressure: constant vector
    W.sub(2)                               # Energy/Level-set: no null space
])
solver = NonlinearVariationalSolver(problem, nullspace=nullsp, ...)
```

## Files in This Folder

### `nse_cavity.py` (Boussinesq Convection)

Solves the Rayleigh-Bernard convection problem in a cavity:

**Features:**
- **Mixed formulation**: $(v, p, e)$ for velocity, pressure, temperature
- **Inf-sup stable elements**: Taylor-Hood (CG2-CG1-CG2)
- **Time stepping**: Backward Euler ($\theta = 1$)
- **Domain**: Unit square $[0,1]^2$
- **Mesh**: $20 \times 20$ structured grid
- **Benchmark parameters**: $\mathrm{Ra} = 10^4$, $\mathrm{Pr} = 0.71$

**Output:**
- VTK file `results/nse_cavity_vpe.pvd` containing velocity, pressure, temperature
- Console output of time steps

**Key variables:**
- `Ra` (Rayleigh number): Controls convection strength
- `Pr` (Prandtl number): Viscosity-to-diffusivity ratio
- `dt` (time step): 0.1
- `t_end` (total simulation time): 30

### `rising_bubble.py` (Multiphase Flow)

Simulates rising bubbles with surface tension using level-set method:

**Features:**
- **Mixed formulation**: $(v, p, \phi)$ for velocity, pressure, level-set
- **Inf-sup stable elements**: Taylor-Hood (CG2-CG1-CG1)
- **Time stepping**: Crank-Nicolson ($\theta = 0.5$)
- **Domain**: Rectangular channel $1.0 \times 2.0$
- **Mesh**: $20 \times 40$ structured grid
- **Two-phase flow**: Different densities and viscosities

**Advanced features:**
- **Level-set reinitialization**: Maintains $|\nabla \phi| = 1$ property
- **Smooth phase transition**: Using smooth sign function to avoid discontinuities
- **Volume conservation monitoring**: Tracks bubble volume to verify accuracy

**Output:**
- VTK file `results/rising_bubble_vpl.pvd` containing velocity, pressure, level-set
- Console output of time steps and bubble volume

**Fluid properties:**
- Phase 1 (upper): $\rho_1 = 100$, $\mu_1 = 10$ (light fluid / air)
- Phase 2 (lower): $\rho_2 = 1000$, $\mu_2 = 1$ (heavy fluid / water)
- Surface tension: $\sigma = 1.0$
- Time step: $\Delta t = 0.1$
- Total time: $T = 20.0$

**Key level-set functions:**
- `Sign(q)`: Smooth sign function avoiding discontinuities
- `rho(l)`: Density field depending on level-set
- `nu(l)`: Viscosity field depending on level-set
- `reinit()`: Level-set reinitialization solver
