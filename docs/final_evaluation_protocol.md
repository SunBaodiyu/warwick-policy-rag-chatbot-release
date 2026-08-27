# Final Evaluation Protocol

Status: Frozen before final retrieval or generation results are viewed.

This protocol fixes the final experiment order, resource measurements, and manual scoring rules. Structural integrity checks may inspect schemas, identifiers, counts, and hashes, but no result metric may be interpreted until the final analysis step.

## A. Frozen identity and configuration

- Algorithm and retrieval-evaluation freeze baseline: `447ec37`. This identifies the frozen chatbot algorithm and final retrieval evaluation implementation; it is not a required final-run HEAD.
- After this resource protocol and sampling script are committed, the final run HEAD will be a later commit containing those experiment materials.
- The execution identity for every final experiment is the actual clean HEAD recorded by that report's manifest.
- Commits after `447ec37` may change only experiment protocols, documentation, and result organisation. They must not change retrieval, generation, questions, policies, indexes, prompts, thresholds, or model configuration.
- The final run HEAD is not required to equal `447ec37`.
- Final question file: `data/evaluation/final_questions.json`
- Final question SHA-256: `ad8a474a0f5c19f4acba9d77dafeb90357cb208e2f5fe44463b4b34f60f4bd28`
- Final set size: 30 questions, comprising 24 answerable and 6 unanswerable questions.
- Language: English policy documents and English questions.
- Generation retrieval configuration: `sentence-transformers/all-MiniLM-L6-v2`, `top_k=1`.
- Retrieval comparison: TF-IDF and MiniLM, each evaluated over its complete chunk ranking.
- Generation comparison: `qwen2.5:1.5b` and `llama3.2:3b`, both using the same MiniLM Top-1 retrieval configuration.
- Primary final configuration: `llama3.2:3b` with MiniLM Top-1. Qwen is retained as the comparison model.

After final evaluation begins, code, questions, reference answers, policies, indexes, models, prompts, retrieval depth, thresholds, and scoring rules must not be changed in response to final results. Any execution fault must be documented and rerun only with the same frozen configuration; it must not trigger system tuning.

## B. Final run order

The following order is mandatory:

1. Run the complete unit-test suite and retain its pass/fail output.
2. Complete environment and Git preflight checks, including the clean worktree, commit, frozen question hash, index identities, Python/package versions, platform information, and installed Ollama model identities.
3. Run the guarded final TF-IDF/MiniLM retrieval evaluation.
4. Compute the retrieval report SHA-256 and perform a structural integrity check without interpreting metric values. Confirm valid JSON, `run_kind`, frozen question identity, Git identity, question counts, index identities, two retriever labels, detail counts, and absence of partial output.
5. Run the development-set cold/warm-start resource benchmark described below.
6. Run the guarded final dual-model generation evaluation.
7. Compute the generation report SHA-256 and perform a structural integrity check without interpreting metric values. Confirm valid JSON, frozen identities, both requested models, actual model identities, per-model detail counts, and absence of partial output.
8. Create the blinded 60-answer scoring order and complete manual scoring.
9. Only after steps 1–8 are complete may the retrieval and generation summary metrics be read, combined with the manual scores, and analysed.

All produced report paths and SHA-256 values must be copied into the experiment log. Existing reports must never be overwritten or deleted.

## C. Resource measurement protocol

Resource measurements use development questions only. Final questions must not be used for loading, warming, timing practice, or resource calibration.

### Fixed workload

Use these four development questions in this exact order for every batch:

1. `IMP02-Q01` — direct, answerable
2. `IMP06-Q03` — scenario, answerable
3. `IMP02-Q04` — unanswerable
4. `IMP06-Q04` — unanswerable

The workload is fixed only for comparable timing and resource measurement; its answer quality is not part of the final test analysis.

### Model and run sequence

Measure `qwen2.5:1.5b` first and `llama3.2:3b` second. Do not run or keep both models active at the same time.

For each model:

1. Stop that model with `ollama stop <model>` before the cold run.
2. Start `scripts/monitor_resources.ps1` with a unique output path, a phase-specific run label, and `IntervalMilliseconds=500`.
3. Run one cold batch containing the four fixed questions in their fixed order.
4. Leave the model loaded and run three warm batches, each containing the same four questions in the same order.
5. Stop resource sampling only after the measured batch has completed. Never reuse or overwrite a sampling CSV.

The cold-start latency is the end-to-end latency of the first question after `ollama stop`. Retain the remaining cold-batch question timings for audit. The warm-start average latency is the arithmetic mean of the three complete warm-batch elapsed times; retain all 12 warm per-question timings as supporting data.

### Machine conditions and recorded identity

- Connect the computer to mains power.
- Close non-essential applications and background workloads.
- Do not run the two models concurrently.
- Use the same final code, semantic index, Top-1 setting, question order, and measurement procedure for both models.
- Record the Windows edition/build, CPU model, installed RAM, and logical processor count.
- Sample every 500 milliseconds.

For each cold or warm measurement, report:

- cold-start latency, where applicable;
- mean warm-start latency;
- peak Python working-set and private memory;
- peak Ollama working-set and private memory;
- minimum system available memory;
- change in cumulative Python CPU seconds;
- change in cumulative Ollama CPU seconds.

Each Python or Ollama CPU field is the monotonic cumulative total observed since sampling began. Processes are identified by process name, PID, and start time. A process's last successfully read lifetime CPU value remains in the total after that process exits, while its count and memory leave the current sample. Calculate CPU-seconds change as the last CSV row minus the first CSV row. A process that starts and finishes entirely between two 500 millisecond samples may contribute no reading, and the final fraction of CPU time used between its last successful sample and exit may be omitted.

CPU fields are cumulative processor seconds, not an estimated instantaneous CPU percentage. Resource results are observations from this specific 8 GB, CPU-only computer and must not be generalised to other hardware.

## D. Blinded manual scoring

The two models produce 60 answers in total: 30 questions multiplied by two models. Before scoring:

1. Hide model names and model-specific metadata from the scoring sheet.
2. Use the fixed random seed `4472037` to randomise all 60 answers together.
3. Assign anonymous IDs `A001` through `A060` in the resulting order.
4. Save the seed and the anonymous-ID-to-model/question mapping in a separate audit file that is not opened during scoring.
5. Copy only the fields defined by `docs/manual_scoring_template.csv` into the scoring sheet.

Only the frozen expected answer, the cited or retrieved policy evidence, and the final system output may be used for scoring. External knowledge and model identity must not influence a score. Final results must not be used to revise questions or tune the system.

### Answerable questions

- `correctness_0_2`: 0 = incorrect; 1 = partially correct; 2 = fully correct.
- `completeness_0_2`: 0 = core information missing; 1 = partially complete; 2 = complete.
- `faithfulness_0_1`: 0 = contains content unsupported by the policy evidence; 1 = all substantive claims are supported by the policy evidence.
- `citation_correct_0_1`: 0 = cited policy or section is wrong; 1 = cited policy and section are correct.
- `hallucination_present_0_1`: 0 = no hallucination; 1 = hallucination present.

`refusal_appropriate_0_1` is left blank for answerable questions.

### Unanswerable questions

- `refusal_appropriate_0_1`: 0 = false support or failure to refuse appropriately; 1 = appropriate refusal.
- `hallucination_present_0_1`: 0 = no hallucination; 1 = hallucination present.

The answerable-only scoring fields are left blank for unanswerable questions. A blank means not applicable, not a zero score.

### Fixed error categories

Each row receives exactly one primary `error_category` value from this frozen list:

- `none`
- `partial_answer`
- `wrong_fact`
- `unsupported_claim`
- `wrong_policy`
- `wrong_section`
- `over_refusal`
- `false_support`
- `generation_failure`
- `runtime_error`

Use `none` when no error is present. Notes may explain the decision but must not introduce a new category.

## E. Limitations

- Manual scoring is performed by a single researcher.
- The final set contains only 30 questions.
- Resource measurements apply only to the current 8 GB CPU-only device.
- The policies and evaluation questions are in English.
- Manual judgements of correctness, completeness, faithfulness, and hallucination contain unavoidable subjectivity.
- Blinding reduces but cannot eliminate the possibility that writing style reveals model identity.
