---
title: Using an LLM as a judge
---
LLM-as-judge means asking a model to grade another model's answer against a rubric. It scales free-text evaluation but has known biases: position bias (preferring the first option in pairwise comparisons), verbosity bias (preferring longer answers) and self-preference (rating its own model family higher).

Reduce bias by swapping the order of candidates and averaging, giving the judge a concrete rubric with examples, asking for a short rationale before the score, and using a different model family as the judge.

Calibrate the judge before trusting it: have humans label 50 to 100 examples and measure agreement, for example with Cohen's kappa. Re-check agreement whenever the judge model or the rubric changes. Prefer deterministic checks wherever they exist and keep the judge for what code cannot measure.
