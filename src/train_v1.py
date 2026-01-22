import os
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp

from dataset import SpaceNetPatchDataset
from metrics import dice_coef, iou_coef

def find_pairs(image_dir, mask_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    mask_paths = []

    for p in image_paths:
        name = os.path.basename(p).replace(".tif", "_mask.tif")
        mp = os.path.join(mask_dir, name)
        if os.path.exists(mp):
            mask_paths.append(mp)
        else:
            # skip images without masks
            mask_paths.append(None)

    pairs = [(im, m) for im, m in zip(image_paths, mask_paths) if m is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]

def main():
    IMAGE_DIR = "data/PS-RGB"
    MASK_DIR  = "data/masks"
    SAVE_DIR  = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_paths, mask_paths = find_pairs(IMAGE_DIR, MASK_DIR)
    print("Pairs:", len(image_paths))

    dataset = SpaceNetPatchDataset(image_paths, mask_paths, patch_size=256, empty_prob=0.15)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    # U-Net (binary segmentation)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # Loss
    loss_fn = smp.losses.DiceLoss(mode="binary")  # good baseline
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    epochs = 5

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)  # (B,H,W)

            optimizer.zero_grad()

            logits = model(images)          # (B,1,H,W)
            probs = torch.sigmoid(logits)

            loss = loss_fn(probs, masks.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # quick eval on a small batch (sanity metric)
        images, masks = next(iter(loader))
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(images))

        dice = dice_coef(probs.cpu(), masks.cpu())
        iou  = iou_coef(probs.cpu(), masks.cpu())

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch} | avg_loss={avg_loss:.4f} | Dice={dice:.4f} | IoU={iou:.4f}")

        ckpt_path = os.path.join(SAVE_DIR, f"unet_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)

if __name__ == "__main__":
    main()
