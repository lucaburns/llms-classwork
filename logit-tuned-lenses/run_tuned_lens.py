import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens import TunedLens


def clean_token(token: str) -> str:
    """
    Make tokens easier to read in terminal output.
    """
    if token == "\n":
        return "[NEWLINE]"

    token = token.replace("\n", "[NEWLINE]")

    if token == "":
        return "[EMPTY]"

    return token


def decode_topk(
    top_probs: torch.Tensor,
    top_indices: torch.Tensor,
    tokenizer: Any,
) -> list[list[list[dict[str, float | str]]]]:
    """
    Convert top-k token IDs and probabilities into a nested Python structure.

    Returns:
        batch_results[batch][position][rank] -> {"token": str, "prob": float}
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


def run_tuned_lens_over_hidden_states(
    lens: TunedLens,
    hidden_states: tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    """
    Run the tuned lens over intermediate hidden states.

    Different libraries/models may expose hidden states with slightly different
    conventions, so this function tries two common alignments:

    Strategy A:
        hidden_states[:-1] with idx = 0..N-1
        (includes embeddings, excludes final hidden state)

    Strategy B:
        hidden_states[1:-1] with idx = 0..N-2
        (skips embeddings, excludes final hidden state)

    Returns:
        A list of tuned-lens logits tensors, one per successfully decoded layer.
    """
    # Strategy A
    try:
        logits_per_layer = []
        for idx, h in enumerate(hidden_states[:-1]):
            logits_per_layer.append(lens(h, idx))
        return logits_per_layer
    except Exception:
        pass

    # Strategy B
    try:
        logits_per_layer = []
        for idx, h in enumerate(hidden_states[1:-1]):
            logits_per_layer.append(lens(h, idx))
        return logits_per_layer
    except Exception as exc:
        raise RuntimeError(
            "Could not align the tuned lens with the model hidden states. "
            "This often means there is no compatible pretrained tuned lens "
            "for the selected model, or the hidden-state indexing convention "
            "does not match."
        ) from exc


def compute_tuned_lens(
    model: Any,
    tokenizer: Any,
    lens: TunedLens,
    prompt: str,
    top_k: int = 5,
    device: str = "cpu",
) -> tuple[list[list[list[dict[str, float | str]]]], list[dict[str, float | str]]]:
    """
    Compute tuned-lens results for the prompt and also return the model's true
    next-token prediction from outputs.logits.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    final_logits = outputs.logits

    tuned_logits_per_layer = run_tuned_lens_over_hidden_states(lens, hidden_states)

    all_layer_results = []
    for logits in tuned_logits_per_layer:
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        decoded = decode_topk(top_probs, top_indices, tokenizer)
        all_layer_results.append(decoded[0])

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


def collect_last_position_probs(
    results: list[list[list[dict[str, float | str]]]],
) -> dict[str, list[float]]:
    """
    Collect probabilities for the last position across tuned-lens layers.
    """
    token_to_probs: dict[str, list[float]] = {}
    num_layers = len(results)

    for layer_idx, layer in enumerate(results):
        last_pos_results = layer[-1]
        for item in last_pos_results:
            token = clean_token(str(item["token"]))
            prob = float(item["prob"])

            if token not in token_to_probs:
                token_to_probs[token] = [0.0] * num_layers

            token_to_probs[token][layer_idx] = prob

    return token_to_probs


def make_line_plot(
    results: list[list[list[dict[str, float | str]]]],
    actual_next_token: list[dict[str, float | str]],
    output_path: str = "tuned_lens_lineplot.png",
    max_tokens: int = 6,
) -> None:
    """
    Create a line plot of token probabilities across tuned-lens layers
    for the last position.
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

    # Ensure every selected token has a probability vector
    for token in selected_tokens:
        if token not in token_to_probs:
            token_to_probs[token] = [0.0] * num_layers

    plt.figure(figsize=(10, 6))
    for token in selected_tokens:
        plt.plot(layers, token_to_probs[token], marker="o", label=token)

    plt.xlabel("Tuned-lens layer")
    plt.ylabel("Probability")
    plt.title("Tuned Lens: Last-Position Token Probabilities Across Layers")
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_heatmap(
    results: list[list[list[dict[str, float | str]]]],
    actual_next_token: list[dict[str, float | str]],
    output_path: str = "tuned_lens_heatmap.png",
    max_tokens: int = 10,
) -> None:
    """
    Create a heatmap of token probabilities across tuned-lens layers
    for the last position.
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

    # Ensure every selected token has a probability vector
    for token in selected_tokens:
        if token not in token_to_probs:
            token_to_probs[token] = [0.0] * num_layers

    heatmap_array = np.array([token_to_probs[token] for token in selected_tokens])

    plt.figure(figsize=(10, max(4, 0.5 * len(selected_tokens))))
    plt.imshow(heatmap_array, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Probability")
    plt.xticks(ticks=list(range(num_layers)), labels=list(range(num_layers)))
    plt.yticks(ticks=list(range(len(selected_tokens))), labels=selected_tokens)
    plt.xlabel("Tuned-lens layer")
    plt.ylabel("Token")
    plt.title("Tuned Lens Heatmap: Last-Position Token Probabilities")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_results(
    results: list[list[list[dict[str, float | str]]]],
    actual_next_token: list[dict[str, float | str]],
) -> None:
    """
    Print all tuned-lens layer/position results and the model's actual next-token prediction.
    """
    for layer_idx, layer in enumerate(results):
        print(f"\nTuned Lens Layer {layer_idx}:")
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
    lens_resource_id: str | None,
    top_k: int,
    results: list[list[list[dict[str, float | str]]]],
    actual_next_token: list[dict[str, float | str]],
) -> None:
    """
    Save tuned-lens results to a JSON file.
    """
    payload = {
        "prompt": prompt,
        "model": model_name,
        "lens_resource_id": lens_resource_id,
        "top_k": top_k,
        "tuned_lens": results,
        "actual_next_token": actual_next_token,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a pretrained tuned lens to inspect layer-wise next-token predictions."
    )
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name")
    parser.add_argument(
        "--lens-resource-id",
        type=str,
        default=None,
        help=(
            "Optional tuned-lens resource ID. "
            "Defaults to the model name/path if omitted."
        ),
    )
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
        default="tuned_lens_lineplot.png",
        help="Path to save the line plot PNG",
    )
    parser.add_argument(
        "--heatmap-output",
        type=str,
        default="tuned_lens_heatmap.png",
        help="Path to save the heatmap PNG",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    try:
        if args.lens_resource_id:
            lens = TunedLens.from_model_and_pretrained(
                model, lens_resource_id=args.lens_resource_id
            )
        else:
            lens = TunedLens.from_model_and_pretrained(model)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load a pretrained tuned lens. "
            "Try a model with a published tuned lens, or supply "
            "--lens-resource-id explicitly."
        ) from exc

    results, actual_next_token = compute_tuned_lens(
        model=model,
        tokenizer=tokenizer,
        lens=lens,
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
            lens_resource_id=args.lens_resource_id,
            top_k=args.top_k,
            results=results,
            actual_next_token=actual_next_token,
        )


if __name__ == "__main__":
    main()
