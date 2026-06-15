"""
========================================================
  Vizualizacija U-Net mreze (encoder + decoder)
========================================================

Propusta jednu sliku kroz model i prikazuje sta mreza
"vidi" na svakom nivou - i u enkoderu (kompresija) i u
dekoderu (rekonstrukcija maske).

  ENCODER: slika -> sve manja, sve apstraktnija
  DECODER: apstraktno -> sve veca, rekonstruise masku
"""

import os
import torch
import rasterio
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from pathlib import Path

# ── Podesavanja ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Unet_model.pth")
SLIKA_PATH = os.path.join(PROJECT_ROOT, "podaci", "test", "satellite", "frankfurt_0000.tif")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "vizualizacija")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Ucitaj model ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Koristim: {device}")

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ── Ucitaj sliku ─────────────────────────────────────────────
with rasterio.open(SLIKA_PATH) as src:
    image = src.read([1, 2, 3]).astype(np.float32) / 255.0

tensor = torch.tensor(image).unsqueeze(0).to(device)
print(f"Ulazna slika: {tensor.shape}")

# ── Hooks za hvatanje decoder izlaza ─────────────────────────
# Decoder se sastoji od vise blokova; kacimo hook na svaki
decoder_izlazi = []

def hook_fn(module, input, output):
    decoder_izlazi.append(output.detach())

# Registruj hook na svaki decoder blok
hooks = []
for blok in model.decoder.blocks:
    h = blok.register_forward_hook(hook_fn)
    hooks.append(h)

# ── Provuci kroz encoder (rucno, da uhvatimo i te nivoe) ─────
with torch.no_grad():
    encoder_izlazi = model.encoder(tensor)
    # Pun forward da se aktiviraju decoder hooks
    finalna_pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

# Skini hooks
for h in hooks:
    h.remove()

print(f"\nEncoder nivoa: {len(encoder_izlazi)}")
for i, f in enumerate(encoder_izlazi):
    print(f"  Enc {i}: {f.shape[1]:>4} kanala, {f.shape[2]}x{f.shape[3]}")

print(f"\nDecoder nivoa: {len(decoder_izlazi)}")
for i, f in enumerate(decoder_izlazi):
    print(f"  Dec {i}: {f.shape[1]:>4} kanala, {f.shape[2]}x{f.shape[3]}")

# ── Vizualizacija: ENCODER prosek po nivou ───────────────────
fig, axes = plt.subplots(1, len(encoder_izlazi), figsize=(4*len(encoder_izlazi), 4))
for i, f in enumerate(encoder_izlazi):
    avg = f[0].mean(dim=0).cpu().numpy()
    axes[i].imshow(avg, cmap='viridis')
    axes[i].set_title(f"Enc {i}\n{f.shape[2]}x{f.shape[3]}")
    axes[i].axis('off')
plt.suptitle("ENCODER - kompresija slike (sve manje, sve apstraktnije)", fontsize=14)
plt.savefig(os.path.join(OUTPUT_DIR, "1_encoder.png"), bbox_inches='tight', dpi=100)
plt.close()
print("\nSacuvano: 1_encoder.png")

# ── Vizualizacija: DECODER prosek po nivou ───────────────────
fig, axes = plt.subplots(1, len(decoder_izlazi), figsize=(4*len(decoder_izlazi), 4))
if len(decoder_izlazi) == 1:
    axes = [axes]
for i, f in enumerate(decoder_izlazi):
    avg = f[0].mean(dim=0).cpu().numpy()
    axes[i].imshow(avg, cmap='magma')
    axes[i].set_title(f"Dec {i}\n{f.shape[2]}x{f.shape[3]}")
    axes[i].axis('off')
plt.suptitle("DECODER - rekonstrukcija maske (sve vece, sve preciznije)", fontsize=14)
plt.savefig(os.path.join(OUTPUT_DIR, "2_decoder.png"), bbox_inches='tight', dpi=100)
plt.close()
print("Sacuvano: 2_decoder.png")

# ── Vizualizacija: ceo U oblik (encoder -> decoder) ──────────
n_enc = len(encoder_izlazi)
n_dec = len(decoder_izlazi)
fig, axes = plt.subplots(2, max(n_enc, n_dec), figsize=(4*max(n_enc, n_dec), 8))

# Gornji red - encoder
for i in range(max(n_enc, n_dec)):
    if i < n_enc:
        avg = encoder_izlazi[i][0].mean(dim=0).cpu().numpy()
        axes[0, i].imshow(avg, cmap='viridis')
        axes[0, i].set_title(f"ENC {i}\n{encoder_izlazi[i].shape[2]}x{encoder_izlazi[i].shape[3]}")
    axes[0, i].axis('off')

# Donji red - decoder
for i in range(max(n_enc, n_dec)):
    if i < n_dec:
        avg = decoder_izlazi[i][0].mean(dim=0).cpu().numpy()
        axes[1, i].imshow(avg, cmap='magma')
        axes[1, i].set_title(f"DEC {i}\n{decoder_izlazi[i].shape[2]}x{decoder_izlazi[i].shape[3]}")
    axes[1, i].axis('off')

plt.suptitle("U-Net: ENCODER (gore) -> DECODER (dole)", fontsize=15)
plt.savefig(os.path.join(OUTPUT_DIR, "3_ceo_unet.png"), bbox_inches='tight', dpi=100)
plt.close()
print("Sacuvano: 3_ceo_unet.png")

# ── Vizualizacija: pojedinacni kanali decodera ───────────────
for dec_idx, f in enumerate(decoder_izlazi):
    n_kanala = min(16, f.shape[1])
    cols = 4
    rows = (n_kanala + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    axes = axes.flatten() if n_kanala > 1 else [axes]
    for k in range(n_kanala):
        axes[k].imshow(f[0, k].cpu().numpy(), cmap='magma')
        axes[k].set_title(f"kanal {k}", fontsize=8)
        axes[k].axis('off')
    for k in range(n_kanala, len(axes)):
        axes[k].axis('off')
    plt.suptitle(f"Decoder nivo {dec_idx} ({f.shape[2]}x{f.shape[3]}) - prvih {n_kanala} kanala", fontsize=12)
    plt.savefig(os.path.join(OUTPUT_DIR, f"4_decoder_nivo{dec_idx}_kanali.png"), bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Sacuvano: 4_decoder_nivo{dec_idx}_kanali.png")

# ── Finalna predikcija ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image.transpose(1, 2, 0))
axes[0].set_title("Ulazna slika")
axes[0].axis('off')
axes[1].imshow(finalna_pred, cmap='gray')
axes[1].set_title("Verovatnoca parka (0-1)")
axes[1].axis('off')
axes[2].imshow((finalna_pred > 0.5), cmap='gray')
axes[2].set_title("Finalna maska (prag 0.5)")
axes[2].axis('off')
plt.suptitle("Od ulaza do finalne maske", fontsize=14)
plt.savefig(os.path.join(OUTPUT_DIR, "5_finalna_predikcija.png"), bbox_inches='tight', dpi=100)
plt.close()
print("Sacuvano: 5_finalna_predikcija.png")

print(f"\nGotovo! Sve vizualizacije u: {OUTPUT_DIR}")