"""
Draws assets/attention_mask.png: the attention mask the model actually builds while generating.
The masks are captured from the first decoder layer during a real `generate()` call on a tiny
randomly initialized model (the mask doesn't depend on the weights).

    python assets/make_attention_figure.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from inference import generate
from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration

IMAGE_TOKEN = 299
N_IMAGE, N_GEN = 16, 9  # 16 image tokens here (256 in PaliGemma-3B-224)
PROMPT = [2, 10, 11, 12, 13, 108]  # <bos> + 4 prompt tokens + "\n"

SURFACE, TEXT, TEXT_2, MASKED = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
IMAGE_C, PROMPT_C, GEN_C = "#2a78d6", "#eb6834", "#1baf7a"

torch.manual_seed(0)
config = PaliGemmaConfig(
    vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                       image_size=56, patch_size=14, num_image_tokens=N_IMAGE),
    text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                     num_attention_heads=4, num_key_value_heads=1, head_dim=16),
    image_token_index=IMAGE_TOKEN, projection_dim=64, hidden_size=64, pad_token_id=0,
)
model = PaliGemmaForConditionalGeneration(config).eval()

# Record the [Q_Len, KV_Len] mask the first decoder layer receives at every forward pass
masks = []
model.language_model.model.layers[0].self_attn.register_forward_pre_hook(
    lambda module, args, kwargs: masks.append(kwargs["attention_mask"][0, 0].clone()), with_kwargs=True
)
input_ids = torch.tensor([[IMAGE_TOKEN] * N_IMAGE + PROMPT])
generate(model, input_ids, torch.randn(1, 3, 56, 56), torch.ones_like(input_ids), max_tokens_to_generate=N_GEN)

# Assemble the full [Tokens, Tokens] picture: one prefill block, then one row per generation step
n_prefix = input_ids.shape[1]
n = n_prefix + N_GEN - 1  # the last generated token is never fed back in
allowed = torch.zeros(n, n, dtype=torch.bool)
allowed[:n_prefix, :n_prefix] = masks[0] == 0
for step, mask in enumerate(masks[1:]):
    allowed[n_prefix + step, : mask.shape[-1]] = mask[0] == 0

# Colour each visible cell by the kind of token being attended to (the key)
key_kind = torch.tensor([1] * N_IMAGE + [2] * len(PROMPT) + [3] * (N_GEN - 1))
cells = torch.where(allowed, key_kind[None, :].expand(n, n), torch.zeros(n, n, dtype=torch.long))

fig, ax = plt.subplots(figsize=(7.2, 7.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.pcolormesh(cells.numpy(), cmap=ListedColormap([MASKED, IMAGE_C, PROMPT_C, GEN_C]), vmin=0, vmax=3,
              edgecolors=SURFACE, linewidth=1.2)
ax.set_xlim(0, n)
ax.set_ylim(n, 0)
ax.set_aspect("equal")
for spine in ax.spines.values():
    spine.set_visible(False)

# Label the three token groups along both axes
groups = [("image tokens", 0, N_IMAGE), ("prompt", N_IMAGE, n_prefix), ("generated", n_prefix, n)]
ax.set_xticks([(a + b) / 2 for _, a, b in groups], [g for g, _, _ in groups], color=TEXT_2, fontsize=10)
ax.set_yticks([(a + b) / 2 for _, a, b in groups], [g for g, _, _ in groups], color=TEXT_2, fontsize=10,
              rotation=90, va="center")
ax.tick_params(length=0, pad=6)
ax.xaxis.set_ticks_position("top")
ax.set_xlabel("attends to (key)", color=TEXT_2, fontsize=10, labelpad=8)
ax.xaxis.set_label_position("top")
ax.set_ylabel("token being processed (query)", color=TEXT_2, fontsize=10, labelpad=8)

fig.legend(
    handles=[Patch(color=c, label=l) for c, l in
             [(IMAGE_C, "image token"), (PROMPT_C, "prompt token"), (GEN_C, "generated token"), (MASKED, "masked")]],
    loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.02),
    handlelength=1.2, handleheight=1.2,
)
fig.suptitle("Prefix-LM attention: the image and prompt see each other,\ngenerated tokens only see the past",
             color=TEXT, fontsize=12.5, y=0.975)
fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.1)
fig.savefig(os.path.join(ROOT, "assets", "attention_mask.png"), dpi=150, facecolor=SURFACE)
print(f"prefill mask {tuple(masks[0].shape)}, {len(masks) - 1} decode masks, figure {n}x{n}")
