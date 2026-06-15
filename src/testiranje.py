import os
import torch
import rasterio
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path

def napravi_model(arhitektura, device):
    """Vraca model na osnovu naziva arhitekture."""
    if arhitektura == 'unet':
        return smp.Unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    elif arhitektura == 'deeplabv3':
        return smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None,
                                 in_channels=3, classes=1).to(device)
    else:
        raise ValueError(f"Nepoznata arhitektura: {arhitektura}. Koristi 'unet' ili 'deeplabv3'.")


def predict_tiles(predikcija_folder, model_path, arhitektura='unet', output_subfolder=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{arhitektura}] Koristim: {device}")

    model = napravi_model(arhitektura, device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    satellite_folder = Path(predikcija_folder) / 'satellite'
    # Ako nije zadat naziv izlaznog foldera, koristi arhitekturu
    if output_subfolder is None:
        output_subfolder = f'prediktovana_maska_{arhitektura}'
    mask_folder = Path(predikcija_folder) / output_subfolder
    mask_folder.mkdir(parents=True, exist_ok=True)

    tiles = sorted(satellite_folder.glob('*.tif'))
    print(f"[{arhitektura}] Pronadjeno {len(tiles)} tile-ova")

    for i, tile_path in enumerate(tiles):
        with rasterio.open(tile_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32) / 255.0
            meta  = src.meta.copy()
            meta.update({'count': 1, 'dtype': 'uint8', 'driver': 'GTiff'})

        tensor = torch.tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(tensor))
            mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        output_path = mask_folder / tile_path.name
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask, 1)

    print(f"[{arhitektura}] Gotovo! Maske sacuvane u: {mask_folder}")
    return str(mask_folder)


if __name__ == '__main__':
    import argparse
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--arhitektura', default='unet', choices=['unet', 'deeplabv3'])
    parser.add_argument('--model', default=None, help='putanja do .pth fajla')
    args = parser.parse_args()

    # Podrazumevane putanje modela
    if args.model is None:
        if args.arhitektura == 'unet':
            args.model = os.path.join(PROJECT_ROOT, "models", "park_model_best.pth")
        else:
            args.model = os.path.join(PROJECT_ROOT, "models", "park_model_deeplabv3_best.pth")

    predikcija_path = os.path.join(PROJECT_ROOT, "podaci", "test")
    predict_tiles(predikcija_path, model_path=args.model, arhitektura=args.arhitektura)