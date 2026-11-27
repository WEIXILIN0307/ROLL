# Verify the restart at each original training step

The eligible-scalar counts are from the resumed Tempo execution. Every point was checked against the saved
checkpoint by original step, metric name, and value. The selected chart values below are rounded for reading.
The linked TSV and JSON retain the values returned by Tempo.

## SFT

Execution `tiny-sft-20261126-123048-df318d50` on ROLL `31a1730`. [Exact TSV](dashboards/sft-readback.tsv) · [Trace samples](dashboards/sft-readback.json)

| Original step | Origin | Matched scalar points | `sft_train/loss` | `time/step_train` |
| ---: | --- | ---: | ---: | ---: |
| 0 | replayed | 12 | 12.0625 | 0.986485 |
| 1 | replayed | 12 | 12.0312 | 0.0700706 |
| 2 | replayed | 17 | 12 | 0.0731935 |
| 3 | new | 12 | 11.75 | 0.846773 |
| 4 | new | 17 | 11.7812 | 0.0724714 |

Total in the resumed execution: **70**.

## Vision RLVR

Execution `tiny-vlm-20261126-123236-5ff01b6b` on ROLL `e8bf333`. [Exact TSV](dashboards/vision-readback.tsv) · [Trace samples](dashboards/vision-readback.json)

| Original step | Origin | Matched scalar points | `time/step_generate` | `actor/lr` |
| ---: | --- | ---: | ---: | ---: |
| 0 | replayed | 221 | 10.3886 | 0.0002 |
| 1 | replayed | 221 | 2.65784 | 0.0001 |
| 2 | replayed | 226 | 2.634 | 0 |
| 3 | new | 221 | 9.89673 | 6e-05 |
| 4 | new | 226 | 3.0696 | 0 |

Total in the resumed execution: **1,115**.

## Vision RLVR zero-valid-response run

In a separate real rollout, the driver skipped actor training for both steps because no response tokens
were valid. After exit, Tempo returned `system/step` for both original steps.

| Step | `system/step` |
| ---: | ---: |
| 0 | 0 |
| 1 | 1 |

[Skip-path readback](dashboards/vision-skip-readback.json)
