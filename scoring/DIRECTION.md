# Research Direction

## Research Problem
The task is active constraint learning from demonstrations: recover an unknown safety boundary in continuous state space while minimizing the number of demonstrations needed.

## Core Contribution
The paper's contribution is uncertainty-guided demonstration selection for constraint inference. Improvements should focus on how a learner extracts constraint information from demonstrations, represents uncertainty over the constraint, and chooses future start/goal queries.

## Approach Scope
Methods should remain in the learning-from-demonstrations and active-query family. A scientist may change the uncertainty model, the boundary representation, the tight-point extraction procedure, or the query optimization strategy, as long as the method still learns constraints from generated demonstrations.

## Out of Bounds
Direct supervised fitting from hidden constraint labels, dense oracle sampling of the obstacle, changing the task into generic binary classification, or increasing the demonstration budget are out of scope. These would bypass the paper's central question rather than improve on it.
