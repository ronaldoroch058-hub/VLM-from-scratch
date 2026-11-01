from typing import Optional
from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


@torch.no_grad()
def generate(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens_to_generate: int,
    stop_token: Optional[int] = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> torch.Tensor:
    """Returns the generated token ids, shape [Batch_Size, Num_Generated]. Stops early only for a batch of 1."""
    kv_cache = KVCache()
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # [Batch_Size, 1]
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if stop_token is not None and next_token.shape[0] == 1 and next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1)

    return torch.cat(generated_tokens, dim=-1)


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)

    # Generate tokens until you see the stop token
    generated_tokens = generate(
        model,
        input_ids=model_inputs["input_ids"],
        pixel_values=model_inputs["pixel_values"],
        attention_mask=model_inputs["attention_mask"],
        max_tokens_to_generate=max_tokens_to_generate,
        stop_token=processor.tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
    dtype: str = "float32",
):
    if model_path is None or prompt is None or image_file_path is None:
        raise ValueError("--model_path, --prompt and --image_file_path are required")
    if dtype not in DTYPES:
        raise ValueError(f"--dtype must be one of {list(DTYPES)}, got {dtype!r}")

    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model ({dtype})")
    model, tokenizer = load_hf_model(model_path, device, DTYPES[dtype])
    model = model.eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
