import os
import glob
import random
import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import SpaceNetPatchDataset
from metrics import dice_coef, iou_coef


def find_pairs(image_dir, mask_dir):
    """
    Match PS-RGB tiles with generated masks.
    Mask filename rule:
        image_xxx.tif -> image_xxx_mask.tif
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    pairs = []

    for img_path in image_paths:
        base = os.path.basename(img_path)
        mask_name = base.replace(".tif", "_mask.tif")
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))

    return pairs


def split_train_val(pairs, val_ratio=0.2, seed=42):
    """
    Split by IMAGE TILE (not patches).
    This prevents leakage between train/val.
    """
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(n_total * val_ratio)

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    return train_pairs, val_pairs


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)

        # metrics.py expects preds as probs in (B,1,H,W) and masks as (B,H,W)
        d = dice_coef(probs.cpu(), masks.cpu())
        j = iou_coef(probs.cpu(), masks.cpu())

        total_dice += d
        total_iou += j
        n_batches += 1

    model.train()
    return total_dice / max(1, n_batches), total_iou / max(1, n_batches)


def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    IMAGE_DIR = "data/PS-RGB"
    MASK_DIR = "data/masks"
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SEED = 42
    VAL_RATIO = 0.2

    PATCH_SIZE = 256
    BATCH_SIZE = 8
    LR = 1e-3
    EPOCHS = 8

    # IMPORTANT: control training time
    # This makes each epoch a fixed number of batches (not huge).
    TRAIN_STEPS_PER_EPOCH = 2000
    VAL_STEPS = 200

    EMPTY_PROB_TRAIN = 0.15
    EMPTY_PROB_VAL = 1.0  # keep empties in val (more honest evaluation)

    # ----------------------------
    # DEVICE
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------------------
    # PAIRS + SPLIT
    # ----------------------------
    pairs = find_pairs(IMAGE_DIR, MASK_DIR)
    print("Pairs:", len(pairs))

    train_pairs, val_pairs = split_train_val(pairs, val_ratio=VAL_RATIO, seed=SEED)
    print(f"Train tiles: {len(train_pairs)} | Val tiles: {len(val_pairs)}")

    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]

    # ----------------------------
    # DATASETS
    # ----------------------------
    train_ds = SpaceNetPatchDataset(
        train_images, train_masks,
        patch_size=PATCH_SIZE,
        empty_prob=EMPTY_PROB_TRAIN
    )

    val_ds = SpaceNetPatchDataset(
        val_images, val_masks,
        patch_size=PATCH_SIZE,
        empty_prob=EMPTY_PROB_VAL
    )

    # ----------------------------
    # LOADERS
    # ----------------------------
    # NOTE: On Windows, if you get crashes -> set num_workers=0
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=True, drop_last=True
    )

    # ----------------------------
    # MODEL
    # ----------------------------
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ----------------------------
    # TRAIN LOOP
    # ----------------------------
    best_val_iou = -1.0
    best_path = os.path.join(SAVE_DIR, "best_model.pt")

    model.train()

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0

        pbar = tqdm(range(TRAIN_STEPS_PER_EPOCH), desc=f"Epoch {epoch}/{EPOCHS}")
        train_iter = iter(train_loader)

        for step in pbar:
            try:
                images, masks = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, masks = next(train_iter)

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            probs = torch.sigmoid(logits)

            loss = loss_fn(probs, masks.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / TRAIN_STEPS_PER_EPOCH

        # ----------------------------
        # VALIDATION
        # ----------------------------
        # limit val batches so it stays fast
        val_iter = iter(val_loader)
        limited_val_batches = []
        for _ in range(VAL_STEPS):
            try:
                limited_val_batches.append(next(val_iter))
            except StopIteration:
                break

        # Build a temporary loader-like list
        # (simple way to limit validation steps)
        model.eval()
        total_dice, total_iou, n = 0.0, 0.0, 0

        with torch.no_grad():
            for images, masks in limited_val_batches:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                probs = torch.sigmoid(logits)

                total_dice += dice_coef(probs.cpu(), masks.cpu())
                total_iou += iou_coef(probs.cpu(), masks.cpu())
                n += 1

        val_dice = total_dice / max(1, n)
        val_iou = total_iou / max(1, n)

        model.train()

        print(f"\nEpoch {epoch}: avg_train_loss={avg_loss:.4f} | val_dice={val_dice:.4f} | val_iou={val_iou:.4f}")

        # ----------------------------
        # SAVE BEST MODEL
        # ----------------------------
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved: {best_path} (val_iou={best_val_iou:.4f})")

    print("\nTraining done.")
    print("Best model:", best_path)
    print("Best val IoU:", best_val_iou)


if __name__ == "__main__":
    main()
