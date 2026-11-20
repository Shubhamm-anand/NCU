# NCU

A personal research toolkit for running Machine Learning Molecular Dynamics (ML-MD) workflows without rewriting glue code every time I switch DFT packages or ML potentials.

![Simulation Workflow](./image.jpg)

## Why I'm Building This

When running atomistic simulations, moving data between DFT codes (like Quantum ESPRESSO or VASP), ML interatomic potentials (MACE, NequIP), and MD integrators usually turns into a pile of one-off scripts. I'm building **NCU** to keep everything under one clean Python interface so I can easily swap calculators, save checkpoints, and customize active-learning loops for my own systems.

## The Physics in Short

Instead of running expensive DFT calculations at every MD step, an ML potential predicts the total energy $E$ as a sum of atomic contributions based on local environments:

$$E = \sum_{i=1}^{N} \varepsilon_i\left(\{\mathbf{r}_{ij}\}_{j \in \mathcal{N}(i)}, Z_i\right)$$

Forces on each atom are then taken directly as the negative gradient of that energy to integrate Newton's equations of motion:

$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E = m_i \frac{d^2 \mathbf{r}_i}{dt^2}$$

## Planned Code Layout

I'm organizing the code around a common `BaseCalculator` interface so the MD loop doesn't care whether forces come from a DFT run or a trained neural network:

```text
NCU/
├── ncu/
│   ├── iodata.py              # Reading/writing NumPy checkpoints (.npz) & text arrays
│   ├── calculators/
│   │   ├── base.py            # BaseCalculator class (energy & force interface)
│   │   ├── dft.py             # Wrappers/parsers for DFT engines (QE, VASP, etc.)
│   │   └── ml_potentials.py   # Wrappers for ML potentials (MACE, NequIP, PyTorch)
│   ├── dynamics/
│   │   ├── integrator.py      # Velocity Verlet & Langevin thermostats
│   │   └── simulation.py      # Main MD loop + periodic checkpoint saving
│   └── workflows/
│       └── active_learning.py # Flag high-uncertainty frames -> DFT -> retrain
├── LICENSE
└── README.md
