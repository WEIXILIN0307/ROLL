"""Generate the tiny BF16 Qwen2, synthetic math data, and initial RLVR config."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--tokenizer-dir", type=Path, help="Use a local copy of the pinned tokenizer and config")
p.add_argument("--project", default="roll-529-draft-bf16")
a = p.parse_args()

root = a.output.resolve()
files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
tokenizer = a.tokenizer_dir or Path(snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
    allow_patterns=files,
))
model = root / "tiny-qwen2"
model.mkdir(parents=True, exist_ok=True)
for file in files[1:]:
    shutil.copyfile(tokenizer / file, model / file)

config = AutoConfig.from_pretrained(tokenizer)
config.update({
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "layer_types": ["full_attention"] * 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "max_position_embeddings": 256, "max_window_layers": 2,
})
torch.manual_seed(123)
AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(model)
dataset = root / "tiny-math.jsonl"
with dataset.open("w") as file:
    for i in range(8):
        file.write(json.dumps({
            "messages": [{"role": "user", "content": f"Compute {i} + 2. Return the answer."}],
            "ground_truth": str(i + 2), "data_source": "openai/gsm8k", "tag": "gsm8k",
        }) + "\n")
template = Path(__file__).with_name("tiny-rlvr.yaml").read_text()
config = OmegaConf.create(template.replace("__RUN__", str(root)))
config.tracker_kwargs.project = a.project
OmegaConf.save(config, root / "first.yaml")
