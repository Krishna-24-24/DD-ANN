# Domain Decomposition Accelerated Neural Networks (DD-ANNs)
### Summer Research Internship Program (SRIP) 2026 — IIT Gandhinagar

**Student:** Krishna (VIT Vellore) , Chitiveli Hemcharan Varma (IIT Gandhinagar)

**Supervisor:** Dr. Abhinav Jha

**Institute:** Indian Institute of Technology Gandhinagar  

---

## Project Overview

This project explores Physics-Informed Neural Networks (PINNs) as 
mesh-free solvers for partial differential equations (PDEs), with 
the goal of combining them with classical domain decomposition 
techniques to build scalable DD-ANN frameworks.

The ultimate application is solving electrostatic models used in 
computational chemistry — specifically the Linearized 
Poisson–Boltzmann (LPB) equation and the COSMO model — which 
arise in biomolecular simulation and solvation energy calculations.

---

## Repository Structure

├── phase1_1D/         # PINN for 1D differential equations

├── phase2_2D/      # PINN for 2D differential equations

├── phase3_3D/      # PINN for 3D differential equations

├── references/        # Key reference papers

└── README.md

---

## Progress

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | PINN for 1D PDEs | ✅ Complete |
| Phase 2 | PINN for 2D PDEs (hard BCs, Fourier features) | ✅ Complete |
| Phase 3 | PINN for 3D PDEs | ✅ Complete |
| Phase 4 | Domain Decomposition in 1D, 2D, 3D | 🔄 In Progress |
| Phase 5 | Application to LPB / COSMO equations | ⏳ Upcoming |

---

## Key Concepts Implemented

- PDE residual minimization via automatic differentiation
- Hard boundary condition enforcement
- Adaptive loss weighting (gradient norm balancing)
- Fourier feature embeddings (spectral bias correction)
- Convergence and ablation studies across 1D, 2D, 3D

---

## Tech Stack

Python · PyTorch · NumPy · Matplotlib · Jupyter Notebook

---

## References

1. Raissi, Perdikaris, Karniadakis — *Physics-Informed Neural Networks* (2019)
2. Wang, Sankaran, Wang, Perdikaris — *An Expert's Guide to Training PINNs* (2023)
