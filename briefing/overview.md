# Overview

This paper studies learning hidden robot safety constraints from demonstrations. Its core result is GP-ACL, which repeatedly learns a GP boundary model from KKT-derived tight points and gradients, samples plausible constraints, and asks for demonstrations that should reduce posterior uncertainty near likely obstacle boundaries.

This workspace implements a scaled 2D obstacle-recovery benchmark that preserves the paper's main computational loop and compares it with the paper's random start/goal query baseline. The main score is safe/unsafe classification accuracy under a fixed demonstration budget.

- **Paper ID:** <!-- e.g. 1706.03762 -->
- **Title:** <!-- e.g. Attention Is All You Need -->
- **Domain:** <!-- e.g. Computer Vision / Object Detection -->
- **TL;DR:** <!-- One sentence: what this paper does and why it matters -->

## Short Summary

<!-- 3-5 sentences covering the problem, approach, and main findings -->

## Key Results

<!-- 2-3 bullet points: main findings or numbers in bullet points -->
