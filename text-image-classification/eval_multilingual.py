import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.helpers import get_data_loaders
from src.models import get_model


def evaluate_on_file(model, data_loader, device):
    """Run evaluation on a single data loader."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            txt, segment, mask, img, label, _ = batch

            txt = txt.to(device)
            segment = segment.to(device)
            mask = mask.to(device)
            img = img.to(device)
            label = label.to(device)

            # Forward pass
            out = model(txt, mask, segment, img)

            # Handle both original (single output) and
            # hybrid (tuple output) models
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            pred    = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total   += label.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",     type=str, required=True)
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--n_classes",   type=int,   default=101)
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--bert_model",  type=str, required=True)
    parser.add_argument("--model",       type=str, default="latefusion")
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--noise_type",  type=str, default="Gaussian")
    parser.add_argument("--batch_sz",    type=int, default=16)
    parser.add_argument("--n_workers",   type=int, default=2)
    parser.add_argument("--hidden_sz",   type=int, default=768)
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg")
    parser.add_argument("--df",          type=int, default=1)
    parser.add_argument("--task",        type=str, default="food101")
    parser.add_argument("--task_type",   type=str, default="classification")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout",     type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # Load trained model
    model = get_model(args)
    checkpoint = torch.load(
        f"{args.savedir}/model_best.pt",
        map_location=device
    )
    
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Results storage
    results = {}

    # Test files to evaluate
    test_files = {
        "English (original)": "test.jsonl",
        "Hindi":              "test_hi.jsonl",
        "Tamil":             "test_ta.jsonl",
        "Bengali":            "test_bn.jsonl",
    }

    noise_levels = [0.0, 5.0, 10.0]

    for lang_name, test_file in test_files.items():
        test_path = f"{args.data_path}/MVSA_Single/{test_file}"

        # Check file exists
        import os
        if not os.path.exists(test_path):
            print(f"Skipping {lang_name} — file not found: {test_path}")
            continue

        print(f"\\nEvaluating on {lang_name}...")
        results[lang_name] = {}

        for noise in noise_levels:
            args.noise_level = noise
            args.test_file   = test_file   # override test file

            # Get data loader for this language + noise
            _, _, test_loaders = get_data_loaders(args)

            # Evaluate
            test_loader = list(test_loaders.values())[0]
            acc = evaluate_on_file(model, test_loader, device)
            results[lang_name][f"e={noise}"] = round(acc * 100, 2)
            print(f"  ε={noise}: {acc*100:.2f}%")

    # Print results table
    print("\\n" + "="*60)
    print("MULTILINGUAL EVALUATION RESULTS")
    print("="*60)
    print(f"{'Language':<25} {'ε=0.0':>8} {'ε=5.0':>8} {'ε=10.0':>8}")
    print("-"*60)
    for lang, scores in results.items():
        e0  = scores.get("e=0.0",  "N/A")
        e5  = scores.get("e=5.0",  "N/A")
        e10 = scores.get("e=10.0", "N/A")
        print(f"{lang:<25} {str(e0):>8} {str(e5):>8} {str(e10):>8}")
    print("="*60)

    # Save results to JSON
    with open(f"{args.savedir}/multilingual_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults saved to {args.savedir}/multilingual_results.json")


if __name__ == "__main__":
    main()