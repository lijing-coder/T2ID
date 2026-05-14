import argparse
import torch
from dataloader import build_eval_dataloader
from model.T2ID import Base_Model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate T2ID ACC.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pth/.pt).")
    parser.add_argument("--meta-csv", required=True, help="Path to metadata CSV.")
    parser.add_argument("--index-csv", required=True, help="Path to evaluation index CSV with an indexes column.")
    parser.add_argument("--image-dir", required=True, help="Image root directory.")
    parser.add_argument("--fundus-col", default="CFP", help="Metadata column containing fundus image paths.")
    parser.add_argument("--oct-col", default="OCT", help="Metadata column containing OCT image paths.")
    parser.add_argument("--label-col", default="MMC_label", help="Metadata column containing class labels.")
    parser.add_argument("--class-names", default="wetAMD,dryAMD,PCV,Normal", help="Comma-separated class names in label-index order.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--p-missing", type=float, default=0.6)
    parser.add_argument("--memory-size", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(args, num_classes):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(args.device).eval()

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    if isinstance(state_dict, torch.nn.Module):
        return state_dict.to(args.device).eval()

    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model = Base_Model(
        num_classes=num_classes,
        p_missing=args.p_missing,
        memory_size=args.memory_size,
        pretrained_backbone=False,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(args.device).eval()


@torch.no_grad()
def evaluate_acc(model, dataloader, device):
    correct = 0
    total = 0
    for fundus, oct_img, targets in dataloader:
        targets = targets.to(device, non_blocking=True)
        logits = model(
            fundus.to(device, non_blocking=True),
            oct_img.to(device, non_blocking=True),
            targets,
            update_memory=False,
        )[0]
        preds = logits.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.numel()
    if total == 0:
        raise ValueError("No samples were loaded for evaluation.")
    return correct / total, correct, total


def main():
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    dataloader = build_eval_dataloader(args, class_names)
    model = load_model(args, num_classes=len(class_names))
    acc, correct, total = evaluate_acc(model, dataloader, args.device)
    print(f"ACC: {acc:.6f}")
    print(f"correct: {correct}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
