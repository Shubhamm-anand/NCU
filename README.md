# NCU — Machine Learning Quantum Molecular Dynamics

**NCU** is a modular Python framework designed to couple **Density Functional Theory (DFT)** engines, **Machine Learning Interatomic Potentials (MLIPs)**, and **Molecular Dynamics (MD)** integrators into a customizable workflow for atomistic simulations and active learning.

![Project Overview](./image.jpg)

## Core Capabilities

* **Unified Potential Interface:** Interchangeable wrappers for multiple DFT software packages (e.g., Quantum ESPRESSO, VASP, CP2K) and equivariant/neural ML potentials (e.g., MACE, NequIP, SchNet).
* **Customizable MD Engine:** Modular time-integrators (Velocity Verlet, Langevin dynamics) with on-the-fly checkpointing.
* **Structured Data I/O (`iodata`):** Fast binary checkpointing (`.npz`) and human-readable trajectory/array export (`.txt`/`.xyz`) for simulation state recovery and training dataset generation.

## Theoretical Background

In ML-accelerated quantum molecular dynamics, a neural network potential approximates the Born–Oppenheimer potential energy surface $E_{\text{DFT}}(\{\mathbf{r}_i, Z_i\})$ as a sum of local atomic energy contributions:

$$E_{\text{ML}} = \sum_{i=1}^{N} \varepsilon_i\left(\{\mathbf{r}_{ij}\}_{j \in \mathcal{N}(i)}, Z_i\right)$$

Atomic forces $\mathbf{F}_i$ are obtained via automatic differentiation of the total predicted energy with respect to nuclear coordinates $\mathbf{r}_i$, driving Newtonian equations of motion:

$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E_{\text{ML}} = m_i \frac{d^2 \mathbf{r}_i}{dt^2}$$

## Planned Code Structure

```text
NCU/
├── ncu/
│   ├── iodata.py              # NumPy checkpoint (.npz) & text array I/O
│   ├── calculators/
│   │   ├── base.py            # BaseCalculator abstract interface
│   │   ├── dft.py             # Multi-code DFT runners & output parsers
│   │   └── ml_potentials.py   # ML interatomic potential wrappers
│   ├── dynamics/
│   │   ├── integrator.py      # Velocity Verlet & thermostat integrators
│   │   └── simulation.py      # MD trajectory driver & checkpoint hooks
│   └── workflows/
│       └── active_learning.py # Uncertainty-driven DFT -> ML retraining loop
├── LICENSE
└── README.md
