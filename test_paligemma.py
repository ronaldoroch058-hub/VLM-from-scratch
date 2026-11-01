"""
Checks the from-scratch PaliGemma against the Hugging Face reference implementation
(transformers 4.45.2, `PaliGemmaForConditionalGeneration`). 4.45 is the first release where HF uses
1-indexed positions in both prefill and decoding: in 4.41-4.44 the decoding positions are 0-indexed, so the
first generated token gets the same position as the last prompt token and the generations diverge after it.

Both models are built from the same tiny config (same architecture as PaliGemma-3B: SigLIP encoder,
linear projector, Gemma decoder with multi-query attention, just much smaller) and share the same
random weights, so no checkpoint download is needed.

    pytest -s test_paligemma.py
"""
import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import PaliGemmaConfig as HFPaliGemmaConfig
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemma
from transformers import SiglipImageProcessor
from transformers.models.paligemma.processing_paligemma import build_string_from_input

from inference import generate
from modeling_gemma import KVCache, PaliGemmaConfig, PaliGemmaForConditionalGeneration
from processing_paligemma import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    add_image_tokens_to_prompt,
    process_images,
)
from utils import load_paligemma

IMAGE_TOKEN, VOCAB, BOS = 299, 300, 2
IMAGE_SIZE, PATCH_SIZE = 56, 14
NUM_IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 16
VISION = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
    image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_image_tokens=NUM_IMAGE_TOKENS, vision_use_head=False,
)
TEXT = dict(
    vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=1, head_dim=16, hidden_activation="gelu_pytorch_tanh",
)
COMMON = dict(image_token_index=IMAGE_TOKEN, projection_dim=64, hidden_size=64, pad_token_id=0)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    hf = HFPaliGemma(
        HFPaliGemmaConfig(vision_config=dict(VISION), text_config=dict(TEXT), attn_implementation="eager", **COMMON)
    ).eval()
    with torch.no_grad():
        # Larger than the default init (which leaves the norms at zero), so that nothing trivially
        # cancels out and the generations aren't the same token repeated
        for p in hf.parameters():
            p.normal_(0, 0.2)
    hf.tie_weights()

    ours = PaliGemmaForConditionalGeneration(
        PaliGemmaConfig(vision_config=dict(VISION), text_config=dict(TEXT), **COMMON)
    ).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)
    ours.tie_weights()
    return hf, ours


def make_inputs(batch_size, prompt_len=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(3, IMAGE_TOKEN, (batch_size, prompt_len), generator=g)
    input_ids = torch.cat(
        [torch.full((batch_size, NUM_IMAGE_TOKENS), IMAGE_TOKEN), torch.full((batch_size, 1), BOS), prompt], dim=1
    )
    pixel_values = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    return input_ids, torch.ones_like(input_ids), pixel_values


def hf_generate(hf, input_ids, attention_mask, pixel_values, n):
    return hf.generate(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
        max_new_tokens=n, do_sample=False, eos_token_id=None, pad_token_id=0,
        output_logits=True, return_dict_in_generate=True,
    )


# ----------------------------------------------------------------------------------------------- model


def test_parameter_names_and_shapes_match_hf(models):
    hf, ours = models
    hf_sd, our_sd = hf.state_dict(), ours.state_dict()
    assert hf_sd.keys() == our_sd.keys()
    for k in hf_sd:
        assert hf_sd[k].shape == our_sd[k].shape, k


@torch.no_grad()
def test_prefill_logits_match_hf(models):
    hf, ours = models
    input_ids, attention_mask, pixel_values = make_inputs(batch_size=2)
    our_logits = ours(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, kv_cache=KVCache())["logits"]
    hf_logits = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask).logits
    print(f"\nprefill: max |logits - HF| = {(our_logits - hf_logits).abs().max():.2e} (max |logit| = {hf_logits.abs().max():.1f})")
    torch.testing.assert_close(our_logits, hf_logits, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_kv_cache_decoding_matches_hf_at_every_step(models):
    hf, ours = models
    input_ids, attention_mask, pixel_values = make_inputs(batch_size=1)
    n = 30
    out = hf_generate(hf, input_ids, attention_mask, pixel_values, n)
    hf_tokens = out.sequences[:, input_ids.shape[1]:]
    assert len(set(hf_tokens[0].tolist())) > 5, "degenerate generation, the test would prove little"

    # Feed HF's tokens through our model with the KV-Cache and compare the logits at every step
    kv_cache, ids, mask, worst = KVCache(), input_ids, attention_mask, 0.0
    for step in range(n):
        logits = ours(input_ids=ids, pixel_values=pixel_values, attention_mask=mask, kv_cache=kv_cache)["logits"][:, -1]
        torch.testing.assert_close(logits, out.logits[step], atol=1e-5, rtol=1e-5)
        worst = max(worst, (logits - out.logits[step]).abs().max().item())
        ids = hf_tokens[:, step : step + 1]
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
    print(f"\ndecoding: max |logits - HF| over {n} steps = {worst:.2e}")


def test_greedy_generation_matches_hf(models):
    hf, ours = models
    input_ids, attention_mask, pixel_values = make_inputs(batch_size=1, seed=1)
    tokens = generate(ours, input_ids, pixel_values, attention_mask, max_tokens_to_generate=30)
    hf_tokens = hf_generate(hf, input_ids, attention_mask, pixel_values, 30).sequences[:, input_ids.shape[1]:]
    assert torch.equal(tokens, hf_tokens)


def test_batched_generation_matches_hf(models):
    # Used to crash: the decode-step position_ids had shape [1, Batch_Size] instead of [Batch_Size, 1]
    hf, ours = models
    input_ids, attention_mask, pixel_values = make_inputs(batch_size=3, seed=2)
    tokens = generate(ours, input_ids, pixel_values, attention_mask, max_tokens_to_generate=10)
    hf_tokens = hf_generate(hf, input_ids, attention_mask, pixel_values, 10).sequences[:, input_ids.shape[1]:]
    assert torch.equal(tokens, hf_tokens)
    assert not torch.equal(tokens[0], tokens[1])  # different images/prompts -> different outputs


def test_vision_tower_runs_once_per_prompt(models):
    _, ours = models
    calls = []
    hook = ours.vision_tower.register_forward_hook(lambda *args: calls.append(1))
    try:
        input_ids, attention_mask, pixel_values = make_inputs(batch_size=1)
        generate(ours, input_ids, pixel_values, attention_mask, max_tokens_to_generate=10)
    finally:
        hook.remove()
    assert len(calls) == 1


# ---------------------------------------------------------------------------------------------- loader


@torch.no_grad()
def test_loader_reads_hf_checkpoint(models, tmp_path):
    hf, _ = models
    hf.save_pretrained(tmp_path, safe_serialization=True)
    model = load_paligemma(str(tmp_path)).eval()
    lm = model.language_model
    assert lm.lm_head.weight is lm.model.embed_tokens.weight  # tied

    input_ids, attention_mask, pixel_values = make_inputs(batch_size=1)
    logits = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, kv_cache=KVCache())["logits"]
    hf_logits = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask).logits
    torch.testing.assert_close(logits, hf_logits, atol=1e-5, rtol=1e-5)


def test_loader_rejects_missing_weights(models, tmp_path):
    hf, _ = models
    hf.save_pretrained(tmp_path, safe_serialization=True)
    (shard,) = tmp_path.glob("*.safetensors")
    tensors = load_file(shard)
    del tensors["language_model.model.layers.0.mlp.up_proj.weight"]
    save_file(tensors, shard, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="up_proj"):
        load_paligemma(str(tmp_path))


@torch.no_grad()
def test_loader_bfloat16(models, tmp_path):
    hf, _ = models
    hf.save_pretrained(tmp_path, safe_serialization=True)
    fp32 = load_paligemma(str(tmp_path)).eval()
    bf16 = load_paligemma(str(tmp_path), dtype=torch.bfloat16).eval()
    assert all(p.dtype == torch.bfloat16 for p in bf16.parameters())
    assert bf16.language_model.model.layers[0].self_attn.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.get_default_dtype() == torch.float32  # restored

    input_ids, attention_mask, pixel_values = make_inputs(batch_size=1)
    ref = fp32(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, kv_cache=KVCache())["logits"]
    low = bf16(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, kv_cache=KVCache())["logits"]
    rel = ((low - ref).abs().max() / ref.abs().max()).item()
    agree = (low.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"\nbfloat16 vs float32: max relative logit error {rel:.3f}, same argmax at {agree:.0%} of positions")
    assert rel < 0.1
    assert agree > 0.8


# ------------------------------------------------------------------------------------------- processor


def our_pixels(image, size=224):
    return process_images(
        [image], size=(size, size), resample=Image.Resampling.BICUBIC, rescale_factor=1 / 255.0,
        image_mean=IMAGENET_STANDARD_MEAN, image_std=IMAGENET_STANDARD_STD,
    )[0]


def hf_pixels(image, size=224):
    processor = SiglipImageProcessor(size={"height": size, "width": size})
    return processor(image, return_tensors="np")["pixel_values"][0]


def test_image_preprocessing_matches_hf():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    np.testing.assert_allclose(our_pixels(image), hf_pixels(image), atol=1e-6)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_non_rgb_images(mode):
    # Used to crash: RGBA gave 4 channels, grayscale and palette images gave a 2D array
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)).convert(mode)
    pixels = our_pixels(image)
    assert pixels.shape == (3, 224, 224)
    # HF's image processor doesn't accept these modes either, so compare with it on the RGB version
    np.testing.assert_allclose(pixels, hf_pixels(image.convert("RGB")), atol=1e-6)


def test_prompt_format_matches_hf():
    ours = add_image_tokens_to_prompt("caption en", "<bos>", 256, "<image>")
    assert ours == build_string_from_input("caption en", "<bos>", 256, "<image>")
    assert ours == "<image>" * 256 + "<bos>caption en\n"
