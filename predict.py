import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from main import build_model


def load_label_map(label_map_path: Path) -> dict:
    with open(label_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def format_label(raw: str) -> str:
    """Return a user-friendly amount name from a raw class label."""
    cleaned = raw.replace("_", " ").replace("-", " ").strip()
    parts = cleaned.split()
    if parts and parts[0].replace("$", "").isdigit():
        val = parts[0].lstrip("$")
        rest = " ".join(parts[1:])
        suffix = f" ({rest})" if rest else ""
        return f"${val} bill{suffix}"
    return cleaned.title()


def make_infer_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


@torch.no_grad()
def predict(
    image_path: Path,
    model_path: Path,
    label_map_path: Path,
    device: str,
    img_size: int,
) -> None:
    device = resolve_device(device)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    label_map = load_label_map(label_map_path)
    transform = make_infer_transform(img_size)

    model = build_model(num_classes=len(label_map))
    # load state dict only (safer pickle behavior)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)
    pred_label = label_map.get(top_idx.item(), str(top_idx.item()))
    pretty = format_label(pred_label)

    print(f"Predicted amount: {pretty}  [class='{pred_label}'] (p={top_prob.item():.4f})")

    top_k = torch.topk(probs, k=min(3, probs.numel()))
    print("\nTop candidates:")
    for prob, idx in zip(top_k.values, top_k.indices):
        label = label_map.get(idx.item(), str(idx.item()))
        print(f"  {format_label(label)}  [class='{label}']: {prob.item():.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Identify USD bill denomination from an image.")
    ap.add_argument("image_path", type=str, help="Path to the bill photo (e.g., bill.jpg).")
    ap.add_argument("--model_path", type=str, default="out_usd_model/best_model.pt",
                    help="Path to the trained model checkpoint.")
    ap.add_argument("--label_map", type=str, default="out_usd_model/label_map.json",
                    help="Path to the label map JSON produced during training.")
    ap.add_argument("--img_size", type=int, default=224, help="Image resize used for inference.")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device to run inference on.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        image_path=Path(args.image_path),
        model_path=Path(args.model_path),
        label_map_path=Path(args.label_map),
        device=args.device,
        img_size=args.img_size,
    )

