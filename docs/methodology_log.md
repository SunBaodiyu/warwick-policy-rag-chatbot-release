# Methodology and Implementation Log

Update this file every time a meaningful design or implementation decision is
made. These notes will support Chapters 3 and 4 of the dissertation.

## Entry template

### Date

YYYY-MM-DD

### Objective

What was the purpose of this work session?

### Action taken

What was implemented or tested?

### Decision and rationale

What option was selected, what alternatives were considered, and why?

### Evidence

- Git commit or changed files:
- Command used:
- Test result:
- Screenshot or result file:

### Problem and resolution

What failed, why did it fail, and how was it resolved?

### Next action

What is the next smallest testable task?

---

## 2026-07-13 - Stage 1 retrieval baseline

### Objective

Create a transparent local retrieval baseline before integrating a language
model.

### Action taken

Implemented local document loading, two chunking strategies, TF-IDF indexing,
command-line search, and automated tests.

### Decision and rationale

TF-IDF was selected as the first baseline because it is lightweight,
interpretable, and suitable for comparison with the later embedding-based RAG
retriever. Policy-aware and fixed-size chunking were both implemented to support
a controlled retrieval experiment.

### Next action

Run the pipeline with two approved public Warwick policies, inspect retrieval
errors, and prepare a small ground-truth question set.

### Iteration evidence

The first run combined several short numbered clauses into one 180-word chunk.
Although the correct source was retrieved, the returned context was broader
than necessary. The policy-aware strategy was therefore revised so that each
numbered clause or heading becomes an independent chunk. Only a single clause
that exceeds the maximum size is divided into overlapping windows. This change
improves citation precision and makes the method easier to explain and compare.

After the revision, the Incident Reporting clause appeared within the top
three TF-IDF results but not at rank one for the wording "account compromise".
This is retained as evidence of a lexical baseline limitation: related word
forms and meanings are not always ranked reliably. The same question will be
reused when evaluating the later embedding-based retriever.
