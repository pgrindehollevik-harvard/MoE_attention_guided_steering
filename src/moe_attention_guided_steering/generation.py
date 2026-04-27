from typing import Any, Dict

from .attention_collection import HFModelResources


def render_plain_prompt(prompt_text: str, plain_template: str = "{prompt}") -> str:
    """Render a plain-text prompt template for non-chat-template models."""
    if "{prompt}" not in plain_template:
        raise ValueError("plain_template must contain a '{prompt}' placeholder.")
    return plain_template.format(prompt=prompt_text)


def encode_prompt_for_generation(
    tokenizer: Any,
    prompt_text: str,
    prompt_format: str = "chat",
    plain_template: str = "{prompt}",
) -> Any:
    """Encode a prompt using either a tokenizer chat template or plain text."""
    if prompt_format == "chat":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("prompt_format='chat' requires tokenizer.apply_chat_template.")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    if prompt_format == "plain":
        return tokenizer(
            render_plain_prompt(prompt_text, plain_template),
            return_tensors="pt",
        )

    raise ValueError("prompt_format must be either 'chat' or 'plain'.")


def normalize_model_inputs(encoded_inputs: Any) -> Dict[str, Any]:
    """Convert tokenizer outputs into a plain mapping for generation."""
    if isinstance(encoded_inputs, dict):
        normalized = dict(encoded_inputs)
    elif hasattr(encoded_inputs, "keys"):
        normalized = {key: encoded_inputs[key] for key in encoded_inputs.keys()}
    else:
        normalized = {"input_ids": encoded_inputs}

    if "input_ids" not in normalized:
        raise KeyError("Tokenizer output must contain input_ids.")
    return normalized


def move_inputs_to_model_device(encoded_inputs: Any, model: Any) -> Any:
    """Move encoded prompt inputs to the model's primary device when possible."""
    model_device = getattr(model, "device", None)
    if model_device is not None and hasattr(encoded_inputs, "to"):
        return encoded_inputs.to(model_device)
    return encoded_inputs


def generate_unsteered_response(
    prompt_text: str,
    resources: HFModelResources,
    prompt_format: str = "chat",
    plain_template: str = "{prompt}",
    max_new_tokens: int = 48,
    min_new_tokens: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_cache: bool = True,
) -> str:
    """Generate one unsteered response from a loaded Hugging Face causal LM."""
    import torch

    tokenizer = resources.tokenizer
    model = resources.model

    encoded_inputs = encode_prompt_for_generation(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        prompt_format=prompt_format,
        plain_template=plain_template,
    )
    encoded_inputs = move_inputs_to_model_device(encoded_inputs, model)
    encoded_inputs = normalize_model_inputs(encoded_inputs)

    generate_kwargs: Dict[str, Any] = {
        "input_ids": encoded_inputs["input_ids"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": use_cache,
    }
    if min_new_tokens > 0:
        generate_kwargs["min_new_tokens"] = min_new_tokens
    if "attention_mask" in encoded_inputs:
        generate_kwargs["attention_mask"] = encoded_inputs["attention_mask"]

    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    prompt_length = encoded_inputs["input_ids"].shape[1]
    response_ids = generated_ids[0, prompt_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()
