---
title: Evaluating retrieval-augmented generation
---
RAG quality has two parts that should be measured separately: retrieval and generation. For retrieval, measure recall@k (did the relevant passage appear in the top k) and precision (how much retrieved text is relevant).

For generation, measure faithfulness or groundedness: every claim in the answer should be supported by the retrieved context. A practical check is citation validation: each citation must point to a passage that was actually retrieved, and an answer without citations is flagged.

Also test the no-answer path: when the knowledge base does not contain the answer, the system should say so instead of inventing one. Keep a small labelled set of questions with the source documents that should be cited.
