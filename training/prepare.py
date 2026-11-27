"""Generate the tiny SFT or Vision RLVR checkpoint/restart fixture."""
import argparse
import json
from pathlib import Path
import shutil

import torch
import yaml
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("kind", choices=("sft", "vision"))
p.add_argument("directory", type=Path)
p.add_argument("--source", type=Path, help="Local copy of the pinned tokenizer/processor files")
a = p.parse_args()
root = a.directory.resolve()
root.mkdir(parents=True, exist_ok=False)
model = root / "model"
model.mkdir()
common = ["config.json", "tokenizer.json", "tokenizer_config.json"]
if a.kind == "sft":
    files = common
    repo = "Qwen/Qwen2.5-0.5B-Instruct"
    rev = "7ae557604adf67be50417f59c2c2f167def9a775"
else:
    files = common + ["preprocessor_config.json", "chat_template.json"]
    repo = "Qwen/Qwen2.5-VL-3B-Instruct"
    rev = "66285546d2b821cf421d4f5eb2576359d3770cd3"
source = a.source or Path(snapshot_download(repo, revision=rev, allow_patterns=files))
for name in files[1:]:
    shutil.copyfile(source / name, model / name)
tiny = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, max_window_layers=2)
if a.kind == "sft":
    config = AutoConfig.from_pretrained(source)
    config.update(dict(tiny, layer_types=["full_attention"] * 2))
    torch.manual_seed(123)
    network = AutoModelForCausalLM.from_config(config)
    for name, count in (("first", 3), ("resumed", 5)):
        rows = [{"instruction": f"What is {i}+1?", "output": str(i+1)} for i in range(count)]
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
else:
    raw = json.loads((source / "config.json").read_text())
    raw.update(tiny, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]})
    raw["vision_config"].update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                                out_hidden_size=64, fullatt_block_indexes=[1], window_size=56)
    torch.manual_seed(42)
    network = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**raw))
    rows = []
    for count in range(1, 9):
        image = Image.new("RGB", (56, 56), "black")
        drawing = ImageDraw.Draw(image)
        for i in range(count):
            x, y = (i % 3) * 16 + 4, (i // 3) * 16 + 4
            drawing.rectangle((x, y, x + 9, y + 9), fill="white")
        image_file = root / f"image-{count}.png"
        image.save(image_file)
        rows.append({"prompt": "<image> Count the white squares. Write the count as a number.",
                     "images": [str(image_file)], "data_source": "mm_math",
                     "reward_model": {"ground_truth": str(count)}})
    (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
network.to(torch.bfloat16).save_pretrained(model)
template = Path(__file__).with_name(f"{a.kind}.yaml").read_text()
first = yaml.safe_load(template.replace("__RUN__", str(root)))
(root / "first.yaml").write_text(yaml.safe_dump(first))
