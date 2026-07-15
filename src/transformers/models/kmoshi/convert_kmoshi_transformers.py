# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assemble a Kmoshi checkpoint: Qwen3-8B temporal backbone + PersonaPlex audio stack.

Sources:
  * Qwen/Qwen3-8B                       -> temporal backbone (`decoder.*`), text
                                           embeddings/lm_head (151936 vocab), QK-norms
  * a converted PersonaPlex-HF folder   -> 16 audio input embeddings, Mimi audio
    (from convert_personaplex_transformers) encoder, depth decoder (dep_q=16)

Weight mapping notes:
  * Kmoshi keeps Moshi's fused GLU: `mlp.fc1 = concat([gate_proj, up_proj])` (gate
    first half - the activated one), `mlp.fc2 = down_proj`. Numerically equivalent
    to Qwen3's 3-matrix SwiGLU.
  * The backbone text embedding is `vocab_size + 1` rows in Moshi-family models
    (last row = the unconditional/text-padding entry used by `generate`). Qwen3 has
    no such row: it is initialized to the mean of the Qwen3 embedding.
  * `depth_decoder.text_embed_tokens` conditions the depth decoder on text tokens of
    the *new* (Qwen3) tokenizer, so the PersonaPlex weights (32001 rows) cannot be
    carried over: it is re-initialized (normal, std=initializer_range) and belongs
    to the 결합부 training stage, together with `input_projections` and the audio
    input embeddings' adaptation.
  * Everything else keeps PersonaPlex weights unchanged (hidden_size 4096 preserved
    across the swap by design).

Resource notes: ~16 GB (Qwen3-8B) download if not cached; peak CPU RAM ~36 GB.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


QWEN3_CHECKPOINT = "Qwen/Qwen3-8B"

# LMGen sampling defaults, inherited unchanged by PersonaPlex (reference lm.py).
TEXT_TEMPERATURE, TEXT_TOP_K = 0.7, 25
AUDIO_TEMPERATURE, AUDIO_TOP_K = 0.8, 250

_RE_QWEN_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# per-layer suffix mapping: Qwen3 -> Kmoshi backbone (`decoder.model.layers.{i}.`)
_LAYER_SUFFIX_MAP = {
    "self_attn.q_proj.weight": "self_attn.q_proj.linear.weight",
    "self_attn.k_proj.weight": "self_attn.k_proj.linear.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.linear.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.linear.weight",
    "self_attn.q_norm.weight": "self_attn.q_norm.weight",
    "self_attn.k_norm.weight": "self_attn.k_norm.weight",
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.down_proj.weight": "mlp.fc2.weight",
}


def map_qwen3_backbone(qwen_sd: dict) -> dict:
    """Maps a Qwen3 `*ForCausalLM` state dict onto Kmoshi backbone (`decoder.*`) keys."""
    out = {}
    gates, ups = {}, {}

    for name, tensor in qwen_sd.items():
        m = _RE_QWEN_LAYER.match(name)
        if m is not None:
            layer, suffix = m.group(1), m.group(2)
            if suffix == "mlp.gate_proj.weight":
                gates[layer] = tensor
            elif suffix == "mlp.up_proj.weight":
                ups[layer] = tensor
            elif suffix in _LAYER_SUFFIX_MAP:
                out[f"decoder.model.layers.{layer}.{_LAYER_SUFFIX_MAP[suffix]}"] = tensor
            else:
                raise ValueError(f"unexpected Qwen3 layer tensor: {name}")
        elif name == "model.embed_tokens.weight":
            # Moshi-family backbone embedding has vocab_size + 1 rows; the extra
            # (unconditional) row is initialized to the mean embedding.
            mean_row = tensor.mean(dim=0, keepdim=True)
            out["decoder.model.embed_tokens.weight"] = torch.cat([tensor, mean_row.to(tensor.dtype)], dim=0)
        elif name == "model.norm.weight":
            out["decoder.model.norm.weight"] = tensor
        elif name == "lm_head.weight":
            out["decoder.lm_head.weight"] = tensor
        else:
            raise ValueError(f"unexpected Qwen3 tensor: {name}")

    if set(gates) != set(ups):
        raise ValueError(f"gate/up layer mismatch: {set(gates) ^ set(ups)}")
    for layer in gates:
        # fused GLU: first half = gate (activated), second half = up (linear)
        out[f"decoder.model.layers.{layer}.mlp.fc1.weight"] = torch.cat([gates[layer], ups[layer]], dim=0)

    return out


def load_personaplex_non_backbone(personaplex_dir: Path) -> dict:
    """Loads every non-`decoder.*` tensor (audio embeddings, Mimi, depth decoder)
    from a converted PersonaPlex-HF folder without materializing the backbone."""
    from safetensors import safe_open

    files = sorted(Path(personaplex_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found under {personaplex_dir}")
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith("decoder."):
                    out[key] = handle.get_tensor(key)
    return out


def convert(qwen3_checkpoint: str, personaplex_dir: Path, out_dir: Path) -> None:
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from transformers.models.kmoshi.configuration_kmoshi import KmoshiConfig
    from transformers.models.kmoshi.modeling_kmoshi import KmoshiForConditionalGeneration

    # KmoshiConfig defaults *are* the Qwen3-8B backbone + PersonaPlex audio stack.
    config = KmoshiConfig()

    print(f"[1/5] loading Qwen3 backbone: {qwen3_checkpoint}", flush=True)
    qwen = Qwen3ForCausalLM.from_pretrained(qwen3_checkpoint, dtype=torch.bfloat16)
    state_dict = map_qwen3_backbone(qwen.state_dict())
    del qwen

    print(f"[2/5] loading PersonaPlex audio stack: {personaplex_dir}", flush=True)
    state_dict.update(load_personaplex_non_backbone(personaplex_dir))

    print("[3/5] materializing the Kmoshi skeleton", flush=True)
    model = KmoshiForConditionalGeneration._from_config(config, dtype=torch.bfloat16)

    # depth_decoder.text_embed_tokens depends on the new (Qwen3) tokenizer vocab:
    # PersonaPlex rows cannot be carried over — re-initialize (결합부 training stage).
    target = model.state_dict()
    reinitialized = []
    for key in list(state_dict.keys()):
        if key in target and state_dict[key].shape != target[key].shape:
            if key == "depth_decoder.text_embed_tokens.weight":
                state_dict[key] = torch.empty_like(target[key]).normal_(std=config.initializer_range)
                reinitialized.append(key)
            else:
                raise ValueError(f"{key}: shape {tuple(state_dict[key].shape)} != expected {tuple(target[key].shape)}")
    print(f"      re-initialized: {reinitialized or 'none'}", flush=True)

    print("[4/5] strict key check + load", flush=True)
    extra_keys = set(state_dict.keys()) - set(target.keys())
    missing_keys = set(target.keys()) - set(state_dict.keys())
    if extra_keys:
        raise ValueError(f"extra keys found: {sorted(extra_keys)[:10]}")
    if missing_keys:
        raise ValueError(f"missing keys: {sorted(missing_keys)[:10]}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    del state_dict

    model.generation_config.do_sample = True
    model.generation_config.temperature = TEXT_TEMPERATURE
    model.generation_config.top_k = TEXT_TOP_K
    model.generation_config.depth_decoder_config = {
        "min_length": config.num_codebooks + 1,
        "max_length": config.num_codebooks + 1,
        "cache_implementation": "static",
        "do_sample": True,
        "temperature": AUDIO_TEMPERATURE,
        "top_k": AUDIO_TOP_K,
    }

    print(f"[5/5] saving to {out_dir} (~19 GB)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    # Kmoshi text streams use the Qwen3 tokenizer (Korean-capable), not Moshi's SPM.
    AutoTokenizer.from_pretrained(qwen3_checkpoint).save_pretrained(str(out_dir))

    print(
        "done. NOTE: this checkpoint needs 결합부 training before it produces speech —\n"
        "the backbone hidden-state distribution changed (Helium -> Qwen3), so the audio\n"
        "input embeddings, depth input_projections and the re-initialized depth text\n"
        "embedding must be realigned first (see the Depformer debate report)."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qwen3", default=QWEN3_CHECKPOINT, help="Qwen3 backbone checkpoint")
    p.add_argument(
        "--personaplex",
        required=True,
        help="path to a converted PersonaPlex-HF folder (from convert_personaplex_transformers)",
    )
    p.add_argument("--out", default="kmoshi-init")
    args = p.parse_args()
    convert(args.qwen3, Path(args.personaplex), Path(args.out))


if __name__ == "__main__":
    main()
