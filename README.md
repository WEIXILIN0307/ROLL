# ROLL #529 · text RLVR checkpoint restart

The screenshot is from Grafana querying Tempo after the real resumed driver exited. The X axis is ROLL’s original `global_step`. Steps 0–2 came from the checkpoint history. Steps 3–4 came from new rollout, reward, and actor-update iterations. The two panels show rollout generation time and actor learning rate, not model quality or a performance comparison.

- [Grafana screenshot](grafana-roll-rlvr.png)
- [Importable Grafana dashboard](grafana-roll-rlvr.json)
- [Tempo readback and checkpoint values](readback.json)

## Reopen or reproduce the figure

In Grafana 13.0.0, configure a Tempo data source with UID `tempo` and its **Tempo query URL** (typically `http://127.0.0.1:3200` when both run on the same host). Import `grafana-roll-rlvr.json`. The bundled dashboard defaults to the recorded project, execution and UTC window. For a new real run, set **Project** and **Execution** to the values from the `RL-Insight experiment:` driver log, and set the dashboard's time picker to cover that run. Tempo selects the spans by execution and metric name. The dashboard sorts by training step, not by event arrival time.

If the original Tempo instance is unavailable, replay the ten recorded values into your own local Tempo OTLP/HTTP receiver. This redraws the panels with the same values and steps while creating new traces and timestamps. It does not rerun training.

```bash
replay=$(python3 replay-readback.py --otlp-url http://127.0.0.1:4318/v1/traces)
python3 make-dashboard.py \
  --project "$(jq -r .project <<< "$replay")" \
  --execution "$(jq -r .execution <<< "$replay")" \
  --output grafana-roll-replayed.json
```

If the receiver uses a different port, use the RL-Insight server's **OTLP traces** port, not the Tempo query port. Import the generated JSON, check that the time picker covers the replay, and refresh if necessary. To capture the loaded dashboard from a locally accessible Grafana instance, use Node 24 and Chrome with `node capture-dashboard.mjs '<dashboard URL>?theme=light&kiosk' figure.png`. If the URL already has a query string, append `&theme=light&kiosk`.

## Re-run training and restart

The run used ROLL `d7a7fdcb5d5bf8eae0bb801f06307e2f5c556104` on one RTX A6000 with Python 3.10, PyTorch 2.11.0+cu130, vLLM 0.20.1, Ray 2.48.0, Transformers 4.57.0, datasets 3.1.0, and tensordict 0.11.0. Both RL-Insight SDK and server were built from `bef6181dcaadc8e643697200c829b875461fefba`, with Tempo 2.6.1. The tiny fixture uses seeded BF16 Qwen2 weights and eight synthetic math prompts. It downloads only the pinned Qwen2.5 tokenizer and model configuration, not pretrained weights.

In a ROLL CUDA environment with the versions above, install the pinned SDK and start the corresponding RL-Insight server with Tempo enabled. [Server setup](https://github.com/verl-project/rl-insight/blob/bef6181dcaadc8e643697200c829b875461fefba/docs/tutorial/server_installation.md)

From the ROLL checkout, pass an unused output directory and select one GPU by numeric CUDA index.

```bash
CUDA_VISIBLE_DEVICES=0 RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080 \
  bash /path/to/evidence/run-training.sh /tmp/roll-529-training
```

The script invokes `examples/start_rlvr_pipeline.py` twice, first through `checkpoint-2`, then from that directory to `checkpoint-4` in a fresh process. Check the second log and select its execution in Grafana. Compare the step and metric values to `pipeline/worker_state_pipeline.json` under `checkpoint-4`. Wall-clock durations vary between runs. The small run executed the actor-update path and 21 saved model tensors changed between checkpoints. The existing FSDP2/DCP path did not preserve AdamW state in this environment, so this does not demonstrate optimizer continuity.
