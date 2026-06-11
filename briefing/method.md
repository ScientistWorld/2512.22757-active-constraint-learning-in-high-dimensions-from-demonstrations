# Paper's Method

## Key Contribution

The paper introduces Gaussian Process Active Constraint Learning (GP-ACL), an iterative demonstration-query algorithm for learning unknown continuous safety constraints from locally optimal demonstrations.

## Approach

At each iteration, the learner extracts tight states and constraint-gradient estimates from demonstrations using KKT optimality conditions. These zero-valued boundary observations and gradient observations train a Gaussian-process representation of the unknown constraint. The learner samples plausible constraint functions from the posterior, finds high-uncertainty points on sampled boundaries, and generates start/goal pairs in directions parallel and orthogonal to the sampled boundary gradient. New demonstrations from those start/goal pairs are added to the dataset.

## Main Claims

The primary claim is that actively selected demonstrations recover unknown nonlinear constraints more accurately than random start/goal selection with the same sparse demonstration budget. Secondary claims are that the method applies to high-dimensional nonlinear dynamics and can handle both simulation and robot hardware settings.
