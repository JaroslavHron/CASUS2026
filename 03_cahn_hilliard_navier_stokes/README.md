# Coupled Cahn-Hilliard and Navier-Stokes: Rising Bubble with Phase Field

A tutorial demonstrating the coupling of the **Cahn-Hilliard equation** with the **Navier-Stokes equations** for multiphase flow simulations. This example uses a phase-field method (concentration field) instead of level-set to track interfaces.

## Overview

This tutorial combines two advanced concepts:

1. **Cahn-Hilliard Equation**: Models phase separation and interface dynamics using a concentration field $c \in [0,1]$
2. **Navier-Stokes Equations**: Governs fluid flow with properties (density, viscosity) that depend on the concentration field

The resulting **coupled system** provides a thermodynamically consistent model for multiphase flow where:
- The interface thickness is automatically regulated by the Cahn-Hilliard dynamics
- Material properties smoothly transition across the interface
- Capillary forces emerge naturally from the free-energy minimization

This is more sophisticated than level-set methods because it preserves mass conservation rigorously and avoids artificial reinitialization.

## Contents

- [Mathematical Formulation](#mathematical-formulation)
- [Phase-Field Representation](#phase-field-representation)
- [Cahn-Hilliard Equation](#cahn-hilliard-equation)
- [Navier-Stokes Coupling](#navier-stokes-coupling)
- [Firedrake Implementation](#firedrake-implementation)
- [Running the Code](#running-the-code)

## Mathematical Formulation

### Complete Coupled System

The coupled Cahn-Hilliard-Navier-Stokes system is:

$$
\begin{aligned}
\frac{\partial c}{\partial t} + \nabla c \cdot \vec{v} &= \nabla \cdot (M \nabla \mu) \quad\text{in } \Omega \\
\mu &= \frac{df}{dc} - \lambda^2 \nabla^2 c \quad\text{in } \Omega \\
\rho(c) \frac{\partial \vec{v}}{\partial t} + \rho(c) (\nabla \vec{v}) \vec{v} - \nabla \cdot \mathbf{T} + \nabla p &= \rho(c) \vec{g} + \vec{f}_{\text{cap}} \quad\text{in } \Omega \\
\nabla \cdot \vec{v} &= 0 \quad\text{in } \Omega
\end{aligned}
$$

where:
- $c$ is the **phase-field (concentration)**, with $c \in [0,1]$ representing the volume fraction
- $\mu$ is the **chemical potential**, derived from the Ginzburg-Landau free energy
- $\rho(c)$ is the **concentration-dependent density**
- $\mathbf{T}$ is the **viscous stress tensor** with concentration-dependent viscosity
- $\vec{f}_{\text{cap}}$ is the **capillary force** arising from interfacial tension
- $M$ is the **mobility** (controls interface dynamics)
- $\lambda$ is the **interface thickness parameter**

## Phase-Field Representation

### Advantages Over Level-Set

| Property | Level-Set | Phase-Field (CH) |
|----------|-----------|------------------|
| **Mass conservation** | Requires careful enforcement | Automatic (built-in) |
| **Reinitialization** | Necessary (can distort interface) | Not needed |
| **Thermodynamics** | Not consistent | Consistent (free-energy stable) |
| **Interface thickness** | Artificial (must choose $\epsilon$) | Physical (determined by $\lambda$) |
| **Capillary forces** | Added artificially | Emerge from energy minimization |

### Concentration-Dependent Properties

**Density:**
$$\rho(c) = \rho_1 + (\rho_2 - \rho_1) c$$

where $\rho_1$ is the light phase (bubble) and $\rho_2$ is the heavy phase (surrounding fluid).

**Viscosity:**
$$\nu(c) = \nu_1 + (\nu_2 - \nu_1) c$$

This provides smooth interpolation between the two fluids across the interface.

## Cahn-Hilliard Equation

### Free Energy Density

The Ginzburg-Landau free energy density is:

$$f(c) = \frac{1}{4}(c^2 - 1)^2$$

This double-well potential has minima at $c = 0$ (light phase) and $c = 1$ (heavy phase), driving phase separation.

### Chemical Potential

The chemical potential derives from the free energy:

$$\mu = \frac{df}{dc} - \lambda^2 \nabla^2 c = (c^3 - c) - \lambda^2 \nabla^2 c$$

The first term is the **bulk chemical potential** (drives phase separation).  
The second term is the **surface energy** (controls interface thickness).

### Transport Equation

The concentration evolves according to:

$$\frac{\partial c}{\partial t} + \vec{v} \cdot \nabla c = M \nabla^2 \mu$$

- **First term (time derivative)**: Change in concentration over time
- **Second term (convection)**: Transport by fluid velocity
- **Third term (diffusion)**: Migration driven by chemical potential gradient

This formulation ensures:
- ✅ **Mass conservation**: $\int_\Omega c \, dx = \text{constant}$ (no diffusion boundary)
- ✅ **Free-energy decay**: System evolves toward lower energy states
- ✅ **Smooth interfaces**: Interface thickness set by $\lambda$

## Navier-Stokes Coupling

### Momentum Equation with Phase-Dependent Properties

$$\rho(c) \frac{\partial \vec{v}}{\partial t} + \rho(c) (\nabla \vec{v}) \vec{v} - \nabla \cdot \mathbf{T} + \nabla p = \rho(c) \vec{g} + \vec{f}_{\text{cap}}$$

where the stress tensor is:

$$\mathbf{T} = 2 \mu(c) \mathbf{D}, \quad \mathbf{D} = \frac{1}{2}(\nabla \vec{v} + \nabla \vec{v}^T)$$

### Capillary Force

The interfacial tension force arises from the chemical potential gradient:

$$\vec{f}_{\text{cap}} = \sigma \nabla c$$

This creates a force at the interface (where $|\nabla c|$ is large) that acts to minimize surface area.

### Continuity Equation

The incompressibility constraint remains unchanged:

$$\nabla \cdot \vec{v} = 0$$

## Firedrake Implementation

### Finite Element Spaces

Mixed function space combining:

```python
Ev = VectorElement("CG", mesh.ufl_cell(), 2)     # Velocity: CG2 vector
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)     # Pressure: CG1
Ec = FiniteElement("CG", mesh.ufl_cell(), 1)     # Concentration: CG1

Evpc = MixedElement([Ev, Ep, Ec])
W = FunctionSpace(mesh, Evpc)
```

This choice ensures:
- **Inf-sup stability** for velocity-pressure (Taylor-Hood pair)
- **Smooth concentration field** with $C^0$ continuity
- **Accurate interface representation** without oscillations

### Weak Formulation

The variational form combines three parts:

**1. Cahn-Hilliard equation:**
$$L_{\text{CH}} = \int_\Omega \left[ \frac{\partial c}{\partial t} \psi + \vec{v} \cdot \nabla c \, \psi + M \nabla \mu \cdot \nabla \psi \right] dx$$

**2. Navier-Stokes with concentration-dependent properties:**
$$L_{\text{NS}} = \int_\Omega \left[ \rho(c) \frac{\partial \vec{v}}{\partial t} \cdot \vec{\psi} + \rho(c) (\nabla \vec{v} \vec{v}) \cdot \vec{\psi} + \mathbf{T} : \nabla \vec{\psi} - \rho(c) \vec{g} \cdot \vec{\psi} - \sigma \nabla c \cdot \vec{\psi} \right] dx$$

**3. Continuity:**
$$L_{\text{cont}} = \int_\Omega \nabla \cdot \vec{v} \, \phi \, dx$$

where $\psi$ and $\phi$ are test functions for concentration and velocity.

### Time Integration with Irksome

Uses **Backward Euler** implicit time-stepping for stability:

```python
from irksome import Dt, TimeStepper, BackwardEuler

scheme = BackwardEuler()
stepper = TimeStepper(L, scheme, t, dt, w, bcs=bcs, ...)

while float(t) < float(T):
    stepper.advance()
```

This automatic time-stepping handles:
- ✅ Implicit treatment of all nonlinear terms
- ✅ Automatic Jacobian computation via `derivative()`
- ✅ PETSc SNES solver for nonlinear systems
- ✅ Consistent with examples 01 and 02

### Boundary Conditions

```python
# No-slip walls (all four boundaries)
bcv = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), [1, 2, 3, 4])

# Concentration: light phase at top, heavy phase at bottom
bcc_top = DirichletBC(W.sub(2), 1.0, [4])
bcc_bot = DirichletBC(W.sub(2), 0.0, [3])

bcs = [bcv, bcc_top, bcc_bot]
```

## Initial Conditions

### Domain and Mesh

- **Domain**: $1.0 \times 2.0$ rectangular region (tall for bubble rise)
- **Mesh**: $20 \times 40$ structured grid

### Initial Concentration Field

Two bubbles (light phase, $c \approx 1$) placed at the top of surrounding heavy fluid ($c \approx 0$):

```python
bubble1 = sqrt((x - 0.5)^2 + (y - 1.5)^2) - 0.25
bubble2 = sqrt((x - 0.3)^2 + (y - 1.3)^2) - 0.20
base = y - 0.5

dist = min(base, min(bubble1, bubble2))
c_init = 0.5 * (1 - tanh(10 * dist))
```

This creates smooth transitions with $c \in [0,1]$.

### Fluid Properties

- **Light phase** (bubbles, $c \to 1$): $\rho_1 = 100$, $\mu_1 = 10$
- **Heavy phase** (base, $c \to 0$): $\rho_2 = 1000$, $\mu_2 = 1$
- **Surface tension**: $\sigma = 1.0$
- **Interface parameter**: $\lambda = 0.01$ (controls interface thickness)
- **Mobility**: $M = 1.0$ (controls interface dynamics speed)

## Running the Code

### On ROSI (with prepared environment)

```bash
source /home/tut10/hron/CASUS2026/setup.sh
cd 03_cahn_hilliard_navier_stokes

# Default run
srun -n 1 -u python 03_ch_ns_rising_bubble.py

# Custom parameters
srun -n 1 -u python 03_ch_ns_rising_bubble.py -dt 0.005 -t_end 15.0 -lambda 0.02 -M 0.5
```

### On Google Colab

```python
%cd 03_cahn_hilliard_navier_stokes
!python 03_ch_ns_rising_bubble.py -dt 0.01 -t_end 10.0
```

### Local Installation

Refer to [Firedrake Installation](https://www.firedrakeproject.org/install.html).

**All dependencies** (Firedrake, Irksome, PETSc) should already be installed if you've set up for examples 01 and 02.

## Command-Line Parameters

Control simulation via PETSc options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-dt` | 0.01 | Time step size |
| `-t_end` | 10.0 | End time |
| `-lambda` | 0.01 | Interface thickness parameter |
| `-M` | 1.0 | Mobility (Cahn-Hilliard diffusivity) |

**Example:**
```bash
python 03_ch_ns_rising_bubble.py -dt 0.005 -lambda 0.005 -M 2.0
```

## Output and Diagnostics

### VTK Output

Writes to `results/ch_nse_rising_bubble_vpc.pvd` containing:
- **velocity**: Flow field $\vec{v}$
- **pressure**: Pressure field $p$
- **concentration**: Phase-field $c$

### Console Monitoring

Each time step prints:
- Current time `t`
- Bubble volume: Volume where $c > 0.5$
- Average concentration: $\int_\Omega c \, dx$ (should be constant for accuracy)

**Example output:**
```
t=0.0000e+00
  volume = 1.2566e+00, c_avg = 6.2500e-01
t=1.0000e-02
  volume = 1.2550e+00, c_avg = 6.2500e-01
t=2.0000e-02
  volume = 1.2528e+00, c_avg = 6.2500e-01
```

Mass conservation is verified by constant $c_{\text{avg}}$.

## Visualization

### ParaView

1. Open `results/ch_nse_rising_bubble_vpc.pvd` in ParaView
2. Use "Warp by Vector" filter to visualize velocity field
3. Use "Contour" filter on concentration to see interface location
4. Animate through time steps to observe bubble dynamics

### Key Observations

- **Interface smoothing**: Phase-field becomes smooth (no sharp jumps)
- **Mass conservation**: Total integral of $c$ remains constant
- **Capillary motion**: Interface evolves to minimize surface area
- **Buoyancy effects**: Lighter bubble rises against gravity
- **Viscous dissipation**: Flow decays over time

## Comparison with Level-Set Method

### Advantages of Phase-Field (Cahn-Hilliard)

✅ **Mass conservation**: Automatic without special treatment  
✅ **No reinitialization**: Interface naturally maintains thickness  
✅ **Thermodynamically consistent**: Based on free-energy minimization  
✅ **Interface dynamics**: Cahn-Hilliard equation models real physics  
✅ **Easier coupling**: Works naturally with other diffusive equations  

### Trade-offs

⚠️ **Computational cost**: Slightly higher (extra concentration equation)  
⚠️ **Nonlinearity**: Cubic polynomial in Cahn-Hilliard equation  
⚠️ **Parameter tuning**: Need to set $\lambda$ and $M$ appropriately  

## Key Concepts Learned

1. **Phase-field methods**: Continuous representation of interfaces
2. **Coupled multiphysics**: How to couple multiple PDE systems
3. **Concentration-dependent properties**: Smooth material property transitions
4. **Free-energy stable schemes**: Energy-decaying time discretizations
5. **Capillary forces**: Interface dynamics from thermodynamics
6. **Mass conservation**: Verification and monitoring in simulations

## Physical Insights

- **Interface thickness**: Controlled by $\lambda$ (larger $\lambda$ = thicker interface)
- **Capillary motion**: Curved interfaces experience forces perpendicular to the interface
- **Buoyancy-driven rise**: Lighter bubble rises due to density contrast
- **Terminal velocity**: Balance between buoyancy and viscous drag
- **Shape evolution**: Interface shape changes to minimize surface tension energy

## References

- [Firedrake Project](https://www.firedrakeproject.org/)
- [Irksome Documentation](https://irksome.readthedocs.io/)
- [PETSc](https://petsc.org/)
- Cahn-Hilliard equation: Classic phase-field model
- Thermodynamically consistent schemes: Shen & Yang (2015) and related work
- Coupled multiphase flow: Recent developments in phase-field methods for fluid dynamics
