# ROLL RL-Insight · SFT and Vision RLVR checkpoint restart

These Grafana screenshots come from the **real resumed GPU executions** in Tempo. Each x-axis uses ROLL’s original training step. Blue bars are steps 0–2 replayed from checkpoint-2. Orange bars are new training steps 3–4 after the worker loaded the saved model. The panels check observability, not performance or model quality.

| Driver | Grafana | Dashboard import | Tempo/checkpoint values |
| --- | --- | --- | --- |
| SFT | [Screenshot](dashboards/sft-grafana.png) | [JSON](dashboards/sft-grafana.json) | [Readback](dashboards/sft-readback.json) |
| Vision RLVR | [Screenshot](dashboards/vision-grafana.png) | [JSON](dashboards/vision-grafana.json) | [Readback](dashboards/vision-readback.json) |

[Step-by-step readback table and exact TSV exports](Readback.md)

## Reproduce the figure

In Grafana 13.0.0, configure a Tempo data source with UID `tempo` and the Tempo **query** URL. Import the dashboard JSON. The saved view uses the recorded project, execution and UTC window. For a new real run, replace **Project** and **Execution** with the values from the driver's `RL-Insight experiment:` line and set the time picker to cover the run. If the new run resumes from a different step, regenerate the dashboard with that step in `--checkpoint-step` so that the colors correctly show the restart boundary.

Without the original Tempo data, replay the ten recorded chart values into your own local Tempo OTLP/HTTP receiver. The replay uses new trace IDs and timestamps and does not run training.

```bash
kind=sft # or vision
replay=$(python3 replay-readback.py --input "dashboards/$kind-readback.json" \
  --otlp-url http://127.0.0.1:4318/v1/traces)
python3 make-dashboard.py --kind "$kind" --uid "roll-$kind-replay" \
  --checkpoint-step 2 \
  --project "$(jq -r .project <<< "$replay")" --execution "$(jq -r .execution <<< "$replay")" \
  --output "$kind-replay.json"
```

Import the generated JSON, covering the replay in the dashboard time picker. Use the **OTLP traces** port for replay and the separate **Tempo query** port in Grafana. You can capture the dashboard from a local Grafana with Node 24 and Chrome using `node capture-dashboard.mjs '<dashboard URL>?theme=light&kiosk' figure.png`. The screenshot tool expects a Grafana view accessible without an interactive login.

## Re-run real training and check all scalars

Use the corresponding ROLL branch. SFT was verified at `31a1730`; Vision RLVR was verified at `e8bf333`. Use a ROLL CUDA environment with Python 3.10, PyTorch 2.11.0+cu130, Ray 2.48.0, vLLM 0.20.1 (Vision RLVR), Transformers 4.57.0, datasets 3.1.0, and tensordict 0.11.0. The worker recipes use one GPU. Install SDK commit `bef6181dcaadc8e643697200c829b875461fefba` and start the matching monitor server with Tempo 2.6.1. [RL-Insight server setup](https://github.com/verl-project/rl-insight/blob/bef6181dcaadc8e643697200c829b875461fefba/docs/tutorial/server_installation.md)

From the corresponding ROLL worktree, choose one GPU by numeric CUDA index and an unused output directory. Set your actual server API and Tempo query ports.

```bash
evidence=/path/to/evidence
kind=sft # or vision
run="/tmp/roll-$kind-restart"
CUDA_VISIBLE_DEVICES=0 RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080 \
  bash "$evidence/training/run-pair.sh" "$kind" "$run"
python "$evidence/training/readback.py" "$run" --tempo-url http://127.0.0.1:3200
```

The harness generates tiny seeded BF16 Qwen2 or Qwen2.5-VL weights. It downloads only pinned tokenizer, processor, and configuration files, not pretrained weights. SFT grows the same synthetic sample prefix from three to five batches. Vision RLVR consumes eight generated images and runs the math reward worker. Each phase uses its own driver process. The resumed process verifies model checkpoint loading and then runs new steps 3–4. Tempo readback compares metric name, value, and original step against both saved `WorkerState` histories. The existing FSDP2/DCP optimizer state limitation remains, so this does not establish AdamW continuity.
