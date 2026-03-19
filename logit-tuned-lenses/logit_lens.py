import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np


def apply_final_norm(model: Any, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Apply the model's final normalization layer, if it exists.

    This makes intermediate hidden states more comparable to the
    representation normally consumed by the lm_head.
    """
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden_state)

    # LLaMA / Mistral style
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden_state)

    return hidden_state


def clean_token(token: str) -> str:
    """
    Make tokens easier to read in terminal output.
    """
    if token == "\n":
        return "[NEWLINE]"

    token = token.replace("\n", "[NEWLINE]")

    # Keep leading spaces visible enough to notice them
    # but avoid raw Python repr output like '\\n'
    if token == "":
        return "[EMPTY]"

    return token

def collect_last_position_probs(results):
    """
    Collect probabilities for the last position across layers.

    Returns:
        token_to_probs: dict mapping token -> list of probabilities by layer
    """
    token_to_probs = {}
    num_layers = len(results)

    for layer_idx, layer in enumerate(results):
        last_pos_results = layer[-1]  # final input position
        for item in last_pos_results:
            token = clean_token(str(item["token"]))
            prob = float(item["prob"])

            if token not in token_to_probs:
                token_to_probs[token] = [0.0] * num_layers

            token_to_probs[token][layer_idx] = prob

    return token_to_probs


def make_line_plot(results, actual_next_token, output_path="logit_lens_lineplot.png", max_tokens=6):
    """
    Create a line plot of token probabilities across layers for the last position.

    Chooses:
    - all tokens in actual_next_token
    - plus any other high-probability tokens seen at the last position
    """
    token_to_probs = collect_last_position_probs(results)
    num_layers = len(results)
    layers = list(range(num_layers))

    selected_tokens = []
    for item in actual_next_token:
        token = clean_token(str(item["token"]))
        if token not in selected_tokens:
            selected_tokens.append(token)

    token_max_probs = sorted(
        token_to_probs.items(),
        key=lambda kv: max(kv[1]),
        reverse=True,
    )

    for token, _ in token_max_probs:
        if token not in selected_tokens:
            selected_tokens.append(token)
        if len(selected_tokens) >= max_tokens:
            break

    plt.figure(figsize=(10, 6))
    for token in selected_tokens:
        plt.plot(layers, token_to_probs[token], marker="o", label=token)

    plt.xlabel("Layer")
    plt.ylabel("Probability")
    plt.title("Logit Lens: Last-Position Token Probabilities Across Layers")
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_heatmap(results, actual_next_token, output_path="logit_lens_heatmap.png", max_tokens=10):
    """
    Create a heatmap of token probabilities across layers for the last position.
    """
    token_to_probs = collect_last_position_probs(results)
    num_layers = len(results)

    selected_tokens = []
    for item in actual_next_token:
        token = clean_token(str(item["token"]))
        if token not in selected_tokens:
            selected_tokens.append(token)

    token_max_probs = sorted(
        token_to_probs.items(),
        key=lambda kv: max(kv[1]),
        reverse=True,
    )

    for token, _ in token_max_probs:
        if token not in selected_tokens:
            selected_tokens.append(token)
        if len(selected_tokens) >= max_tokens:
            break

    heatmap_array = np.array([token_to_probs[token] for token in selected_tokens])

    plt.figure(figsize=(10, max(4, 0.5 * len(selected_tokens))))
    plt.imshow(heatmap_array, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Probability")
    plt.xticks(ticks=list(range(num_layers)), labels=list(range(num_layers)))
    plt.yticks(ticks=list(range(len(selected_tokens))), labels=selected_tokens)
    plt.xlabel("Layer")
    plt.ylabel("Token")
    plt.title("Logit Lens Heatmap: Last-Position Token Probabilities")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def decode_topk(
    top_probs: torch.Tensor,
    top_indices: torch.Tensor,
    tokenizer: Any,
    batch_index: int = 0,
) -> list[list[list[dict[str, float | str]]]]:
    """
    Convert top-k token IDs and probabilities into a nested Python structure.

    Returns:
        decoded[position][rank] -> {"token": str, "prob": float}
        wrapped in an outer batch dimension for consistency
    """
    batch_results = []

    for b in range(top_indices.shape[0]):
        seq_results = []
        for pos in range(top_indices.shape[1]):
            pos_results = []
            for k in range(top_indices.shape[2]):
                token_id = top_indices[b, pos, k].item()
                token_str = tokenizer.decode([token_id])
                prob = top_probs[b, pos, k].item()
                pos_results.append({"token": token_str, "prob": prob})
            seq_results.append(pos_results)
        batch_results.append(seq_results)

    return batch_results


def compute_logit_lens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    top_k: int = 5,
    device: str = "cpu",
) -> tuple[list[list[list[dict[str, float | str]]]], list[dict[str, float | str]]]:
    """
    Compute vanilla logit lens results for all hidden states and also return the
    model's true next-token prediction from outputs.logits.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    final_logits = outputs.logits

    all_layer_results = []

    for hidden_state in hidden_states:
        normalized_hidden = apply_final_norm(model, hidden_state)
        logits = model.lm_head(normalized_hidden)

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        decoded = decode_topk(top_probs, top_indices, tokenizer)
        # decoded has shape [batch][position][rank]; batch is 1 here
        all_layer_results.append(decoded[0])

    # Actual next-token prediction from the model's final logits
    final_probs = torch.softmax(final_logits, dim=-1)
    final_top_probs, final_top_indices = torch.topk(final_probs, top_k, dim=-1)

    last_position = final_top_indices.shape[1] - 1
    actual_next_token = []

    for k in range(top_k):
        token_id = final_top_indices[0, last_position, k].item()
        token_str = tokenizer.decode([token_id])
        prob = final_top_probs[0, last_position, k].item()
        actual_next_token.append({"token": token_str, "prob": prob})

    return all_layer_results, actual_next_token


def print_results(
    results: list[list[list[dict[str, float | str]]]],
    actual_next_token: list[dict[str, float | str]],
) -> None:
    """
    Print all layer/position results and the model's actual next-token prediction.
    """
    for layer_idx, layer in enumerate(results):
        print(f"\nLayer {layer_idx}:")
        for pos_idx, pos_results in enumerate(layer):
            print(f"  Position {pos_idx}:")
            for item in pos_results:
                token = clean_token(str(item["token"]))
                prob = float(item["prob"])
                print(f"    {token}: {prob:.4f}")

    print("\nActual model prediction for the next token:")
    for item in actual_next_token:
        token = clean_token(str(item["token"]))
        prob = float(item["prob"])
        print(f"  {token}: {prob:.4f}")


def save_results(
    path: str,
    prompt: str,
    model_name: str,
    top_k: int,
    results: list[list[list[dict[str, float | str]]]],
    actual_next_token: list[dict[str, float | str]],
) -> None:
    """
    Save results to a JSON file.
    """
    payload = {
        "prompt": prompt,
        "model": model_name,
        "top_k": top_k,
        "logit_lens": results,
        "actual_next_token": actual_next_token,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla logit lens for causal language models.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tokens to show")
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to save JSON results",
    )
    parser.add_argument(
        "--lineplot-output",
        type=str,
        default="logit_lens_lineplot.png",
        help="Path to save the line plot PNG",
    )
    parser.add_argument(
        "--heatmap-output",
        type=str,
        default="logit_lens_heatmap.png",
        help="Path to save the heatmap PNG",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    results, actual_next_token = compute_logit_lens(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        top_k=args.top_k,
        device=device,
    )

    print_results(results, actual_next_token)

    make_line_plot(
        results=results,
        actual_next_token=actual_next_token,
        output_path=args.lineplot_output,
    )

    make_heatmap(
        results=results,
        actual_next_token=actual_next_token,
        output_path=args.heatmap_output,
    )

    print(f"\nSaved line plot to: {args.lineplot_output}")
    print(f"Saved heatmap to: {args.heatmap_output}")

    if args.json_output is not None:
        save_results(
            path=args.json_output,
            prompt=args.prompt,
            model_name=args.model,
            top_k=args.top_k,
            results=results,
            actual_next_token=actual_next_token,
        )


if __name__ == "__main__":
    main()