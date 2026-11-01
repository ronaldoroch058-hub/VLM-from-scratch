from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
import torch


def load_hf_model(
    model_path: str, device: str, dtype: torch.dtype = torch.float32
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    model = load_paligemma(model_path, device, dtype)

    return (model, tokenizer)


def load_paligemma(
    model_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> PaliGemmaForConditionalGeneration:
    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model directly on the target device and in the target dtype, so a float32 copy of the
    # whole model never has to fit in memory when we want bfloat16 / float16.
    # (The RoPE inverse frequencies are created with an explicit .float(), so they stay in float32.)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = PaliGemmaForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)

    # The checkpoint doesn't store the LM head: it shares its weights with the token embeddings
    model.tie_weights()

    # Find all the *.safetensors files
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No *.safetensors files found in {model_path}")

    # Copy the tensors into the model one at a time, so peak memory is the model plus a single tensor
    # (instead of the model plus the whole checkpoint)
    state_dict = model.state_dict()
    loaded, unexpected = set(), []
    with torch.no_grad():
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in state_dict:
                        unexpected.append(key)
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.shape != state_dict[key].shape:
                        raise ValueError(
                            f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)}, model {tuple(state_dict[key].shape)}"
                        )
                    state_dict[key].copy_(tensor)
                    loaded.add(key)

    # Fail loudly instead of silently running with randomly initialized weights
    missing = set(state_dict) - loaded - {"language_model.lm_head.weight"}
    if missing:
        raise ValueError(f"Missing weights in the checkpoint: {sorted(missing)}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} checkpoint tensors not used by the model: {unexpected[:5]}...")

    return model
