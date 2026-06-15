import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import albumentations as A
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import os

# ── Dataset ───────────────────────────────────────────────────────────────────

class ParkDataset(torch.utils.data.Dataset):
    def __init__(self, tiles_folder, transform=None):
        self.satellite_folder = Path(tiles_folder) / 'satellite'
        self.mask_folder      = Path(tiles_folder) / 'mask'
        self.transform        = transform

        csv_path  = Path(tiles_folder) / 'tile_map.csv'
        df        = pd.read_csv(csv_path, header=0, names=['tile', 'original', 'city'])
        self.pairs = [(row['tile'], row['tile']) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tile_name, mask_name = self.pairs[idx]

        with rasterio.open(self.satellite_folder / tile_name) as src:
            image = src.read([1, 2, 3]).astype(np.float32) / 255.0

        with rasterio.open(self.mask_folder / mask_name) as src:
            mask = src.read(1).astype(np.float32)

        if self.transform:
            aug   = self.transform(image=image.transpose(1,2,0), mask=mask)
            image = aug['image'].transpose(2,0,1)
            mask  = aug['mask']

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)

# ── Augmentacija ──────────────────────────────────────────────────────────────

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    tiles_path = os.path.join(PROJECT_ROOT, "podaci", "tiles_novi")
    model_out  = os.path.join(PROJECT_ROOT, "models/Unet_model.pth")

    dataset = ParkDataset(tiles_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    print(f"Trening: {len(train_ds)} slika | Validacija: {len(val_ds)} slika")

    images, masks = next(iter(train_loader))
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Image min/max: {images.min():.3f} / {images.max():.3f}")
    print(f"Mask min/max: {masks.min():.3f} / {masks.max():.3f}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Koristim: {device}")

    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = smp.losses.DiceLoss(mode='binary')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(50):
        # -- Trening --
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # -- Validacija --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_loss += criterion(model(images), masks).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/50 | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_out)
            print(f"           ✓ Novi best model sacuvan (val_loss={val_loss:.4f})")

    print("Trening zavrsen!")