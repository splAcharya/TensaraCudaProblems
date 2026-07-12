# Results and Evidence Standard

Structured evidence is authoritative. Markdown reports are human-facing views
and must not promote compilation, sanitizer, or benchmark-only rows to
correctness evidence.

## Repository index

`RESULTS_INDEX.md` summarizes coverage and freshness. Its generated cells come
from validated evidence; recommendations and owner notes remain human-authored.

## Per-problem report order

1. Executive Summary
2. Decision Table
3. Key Observations
4. Correctness Evidence
5. Performance Evidence
6. Recommendation
7. Next Validation Steps
8. Appendix A - Test Inventory
9. Appendix B - Kernel Variants
10. Appendix C - Launch Sweeps
11. Appendix D - Raw Evidence Index
12. Appendix E - Reproduction Metadata
13. Appendix F - Known Limitations

Generated tables and human interpretation must occupy separate sections.
Generators must fail on malformed evidence and must never overwrite historical
raw logs or human conclusions.

## Evidence claims

- Compile: builds under the recorded toolchain.
- Exact fixture: matches the named hand-authored case.
- CPU reference: agrees with the owner's reference within existing tolerance.
- Metamorphic: satisfies only the named property.
- Sanitizer: selected tooling found no issue in the selected run.
- Benchmark unverified: timing only.
- Legacy partial: historical evidence with incomplete provenance.

Every generated observation must link to a stable evidence ID. Semantic case
IDs remain stable across runs; run IDs and evidence digests distinguish actual
executions.
