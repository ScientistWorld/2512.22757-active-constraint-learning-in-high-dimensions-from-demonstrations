# Paper's Method

## Key Contribution

The paper introduces Gaussian Process Active Constraint Learning (GP-ACL), an iterative demonstration-query algorithm for learning unknown continuous safety constraints from locally optimal demonstrations.

## Approach

At each iteration, the learner extracts tight states and constraint-gradient estimates from demonstrations using KKT optimality conditions. These zero-valued boundary observations and gradient observations train a Gaussian-process representation of the unknown constraint. The learner samples plausible constraint functions from the posterior, finds high-uncertainty points on sampled boundaries, and generates start/goal pairs in directions parallel and orthogonal to the sampled boundary gradient. New demonstrations from those start/goal pairs are added to the dataset.

## Main Claims

The primary claim is that actively selected demonstrations recover unknown nonlinear constraints more accurately than random start/goal selection with the same sparse demonstration budget. Secondary claims are that the method applies to high-dimensional nonlinear dynamics and can handle both simulation and robot hardware settings.

## Packaged specialization

The implementation uses the paper's double-integrator setting at dt=1 and T=30 on the paper's [-15,15]^2 workspace. With free velocities/controls and position-only endpoints, eliminating dynamics is exact: successive position differences determine velocities, and successive velocity differences determine controls. It solves the reduced tightness/gradient KKT linear programs, keeps each gradient estimate as a unit direction (the estimates are identified only up to a positive scale), and trains a value/derivative GP with the loss of the GP constraint-learning work the paper cites for its training (negative marginal likelihood plus a hinge keeping demonstrated states predicted safe). The constant prior mean is tied to -0.1 x signal, so unexplored states keep prior safety probability Phi(0.1)=0.54, just below the paper's endpoint level beta=0.55. The paper's own posterior is zero-mean; that setting was tested as the reference and rejected on the development split (the fitted lengthscale ran to its bound and uniform accuracy fell to 0.73). It samples the conditioned posterior with 1,000 features and alpha 0.3. Problem 3 is solved over the sampled boundary within one fitted lengthscale of the observed tight states (zero-crossing enumeration plus continuous polishing); Problems 4-6 use continuous scalar root solves with Eq. 10 read as a safe probability. Alg. 1 is iterated, refitting after every batch of P=4 posterior draws, until the demonstration budget is spent; repeated start/goal requests are skipped. See HARDENING.md for unspecified settings and deviations.
