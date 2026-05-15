import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

from dataloader import build_dataloader
from model.T2ID import T2ID


def parse_args():
    parser = argparse.ArgumentParser(description="Train T2ID and report validation ACC each epoch.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for resume/eval. Optional in train mode, required in eval mode.")
    parser.add_argument("--meta-csv", required=True, help="Path to metadata CSV.")
    parser.add_argument("--image-dir", required=True, help="Image root directory.")
    parser.add_argument("--train-index-csv", default=None, help="Training index CSV with an indexes column.")
    parser.add_argument("--val-index-csv", default=None, help="Validation/test index CSV with an indexes column.")
    parser.add_argument("--index-csv", default=None, help="Alias for --val-index-csv in eval mode.")
    parser.add_argument("--fundus-col", default="CFP", help="Metadata column containing fundus image paths.")
    parser.add_argument("--oct-col", default="OCT", help="Metadata column containing OCT image paths.")
    parser.add_argument("--label-col", default="MMC_label", help="Metadata column containing class labels.")
    parser.add_argument("--class-names", default="wetAMD,dryAMD,PCV,Normal", help="Comma-separated class names in label-index order.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--p-missing", type=float, default=0.6)
    parser.add_argument("--memory-size", type=int, default=500)
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use torchvision pretrained ResNet backbones when starting training from scratch.")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def require_path(value, name):
    if value is None:
        raise ValueError(f"{name} is required.")
    return value


def build_model(args, num_classes):
    model = T2ID(
        num_classes=num_classes,
        p_missing=args.p_missing,
        memory_size=args.memory_size,
        pretrained_backbone=args.pretrained_backbone,
    )
    return model.to(args.device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    if isinstance(state_dict, torch.nn.Module):
        return state_dict.to(device)

    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def forward_logits(model, fundus, oct_img, targets, update_memory):
    outputs = model(fundus, oct_img, targets, update_memory=update_memory)
    logits = outputs[0]
    confidence_loss = outputs[2]
    return logits, confidence_loss


def batch_acc(logits, targets):
    return logits.argmax(dim=1).eq(targets).sum().item()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for fundus, oct_img, targets in dataloader:
        fundus = fundus.to(device, non_blocking=True)
        oct_img = oct_img.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, confidence_loss = forward_logits(model, fundus, oct_img, targets, update_memory=True)
        loss = criterion(logits, targets) + confidence_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += batch_acc(logits, targets)
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_acc(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for fundus, oct_img, targets in dataloader:
        fundus = fundus.to(device, non_blocking=True)
        oct_img = oct_img.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, confidence_loss = forward_logits(model, fundus, oct_img, targets, update_memory=False)
        loss = criterion(logits, targets) + confidence_loss

        total_loss += loss.item() * targets.size(0)
        correct += batch_acc(logits, targets)
        total += targets.size(0)

    if total == 0:
        raise ValueError("No samples were loaded for evaluation.")
    return total_loss / total, correct / total, correct, total


def save_checkpoint(model, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def main():
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    val_index_csv = args.val_index_csv or args.index_csv
    require_path(val_index_csv, "--val-index-csv or --index-csv")

    if args.mode == "eval":
        require_path(args.checkpoint, "--checkpoint")

    model = build_model(args, num_classes=len(class_names))
    if args.checkpoint is not None:
        model = load_checkpoint(model, args.checkpoint, args.device)

    criterion = nn.CrossEntropyLoss()
    val_loader = build_dataloader(args, val_index_csv, class_names, is_train=False)

    if args.mode == "eval":
        val_loss, val_acc, correct, total = evaluate_acc(model, val_loader, criterion, args.device)
        print(f"Eval Loss: {val_loss:.4f}, Eval ACC: {val_acc:.4f} ({correct}/{total})")
        return

    train_index_csv = require_path(args.train_index_csv, "--train-index-csv")
    train_loader = build_dataloader(args, train_index_csv, class_names, is_train=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    best_path = Path(args.save_dir) / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc, correct, total = evaluate_acc(model, val_loader, criterion, args.device)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, best_path)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train ACC: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f} ({correct}/{total}) | "
            f"Best ACC: {best_acc:.4f}"
        )

if __name__ == "__main__":
    main()
