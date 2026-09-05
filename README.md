# Interpretable-PINN-Bottleneck

**Physics-Informed Neural Networks with Concept Bottleneck Layers:** Disentangling Known Physics, Unmodeled Dynamics, and Residuals for Physical Steering & Faithfulness.

*<img width="578" height="152" alt="image" src="https://github.com/user-attachments/assets/25e46d08-5653-4632-9ca9-276d6732ba22" />*

**Concept Bottleneck PINN for 1D Burgers' Equation:** Mechanistic Interpretability in Continuous Physical Systems (Inspired by *Scaling Inherently Interpretable Language Models*, Guide Labs)

The Concept Bottleneck PINN successfully decoupled the 1D Burgers' equation into Convection (49.29%) and Diffusion (47.71%). Most importantly, Unknown Dynamics were strictly suppressed to 0.00%.

**Code Navigation:**
* `loop.py`: Curriculum learning loop
* `loss.py`: PINN custom loss code
* `models.py`: Bottleneck architecture code
* `pinn.py`: PINN engine code
