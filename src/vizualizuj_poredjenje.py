"""
========================================================
  Vizualizacija poredjenja UNet i DeepLabV3+ predikcija
========================================================

Za 20 nasumicnih primera generise 2-redna slika:
  Red 1 — Satelit + prava maska | UNet overlay | DeepLabV3+ overlay
  Red 2 — Satelit (referenca)   | UNet razlika | DeepLabV3+ razlika

Izlaz: vizualizacija_poredjenja/primer_XX_frankfurt_XXXX.png
"""

import os
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Putanje ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_DIR   = Path(PROJECT_ROOT) / "test"
SAT_DIR    = TEST_DIR / "satellite"
MASK_DIR   = TEST_DIR / "mask"
UNET_DIR   = TEST_DIR / "prediktovana_maska_unet"
DLV3_DIR   = TEST_DIR / "prediktovana_maska_deeplabv3"
OUTPUT_DIR = Path(PROJECT_ROOT) / "vizualizacija_poredjenja"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PRIMERI    = 20
SEED         = 42
OPACITY_MASK = 0.67   # koliko je crni overlay neproziran za non-park piksele
OPACITY_DIFF = 0.60   # koliko je crveni overlay neproziran za razlike

# ── Pomocne funkcije ──────────────────────────────────────────────────────────

def read_as_rgb(path: Path) -> np.ndarray:
    """Cita TIF fajl, vraca float32 RGB (H, W, 3) normalizovan na [0, 1]."""
    with rasterio.open(path) as src:
        data = src.read()   # (C, H, W)

    if data.shape[0] == 1:
        band = data[0].astype(np.float32)
        band = (band - band.min()) / (band.max() - band.min() + 1e-8)
        return np.stack([band, band, band], axis=-1)

    rgb = data[:3].astype(np.float32)          # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))         # (H, W, 3)
    for c in range(3):
        ch = rgb[:, :, c]
        rgb[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return rgb


def read_mask(path: Path) -> np.ndarray:
    """Cita binarnu masku, vraca (H, W) uint8 array (0 ili 1)."""
    with rasterio.open(path) as src:
        raw = src.read(1)
    return (raw > 0).astype(np.uint8)


def mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Park pikseli  →  originalna satelitska boja (100% vidljivi).
    Non-park pikseli → crni overlay s OPACITY_MASK neprozirnoscu.
    """
    result = rgb.copy()
    non_park = mask == 0
    result[non_park] *= (1.0 - OPACITY_MASK)   # = satellite * 0.33
    return np.clip(result, 0, 1)


def diff_overlay(rgb: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Pikseli gde predikcija != prava maska dobijaju crveni overlay s OPACITY_DIFF.
    """
    result = rgb.copy()
    razlika = (pred_mask != true_mask)
    result[razlika] = result[razlika] * (1.0 - OPACITY_DIFF) + \
                      np.array([1.0, 0.0, 0.0]) * OPACITY_DIFF
    return np.clip(result, 0, 1)


def postoci_park(mask: np.ndarray) -> float:
    return mask.sum() / mask.size * 100


# ── Odaberi nasumicne primere ─────────────────────────────────────────────────

all_tiles = sorted(SAT_DIR.glob("*.tif"))
if not all_tiles:
    raise FileNotFoundError(f"Nisu pronadjeni TIF fajlovi u: {SAT_DIR}")

random.seed(SEED)
odabrani = random.sample(all_tiles, min(N_PRIMERI, len(all_tiles)))

print(f"Generisujem {len(odabrani)} vizualizacija...\n")

# ── Glavna petlja ─────────────────────────────────────────────────────────────

sacuvano = 0

for i, sat_path in enumerate(odabrani):
    stem = sat_path.stem    # npr. "frankfurt_0042"

    mask_path = MASK_DIR / f"{stem}.tif"
    unet_path = UNET_DIR / f"{stem}.tif"
    dlv3_path = DLV3_DIR / f"{stem}.tif"

    if not all(p.exists() for p in [mask_path, unet_path, dlv3_path]):
        print(f"  [!] Preskacam {stem} — nedostaje fajl.")
        continue

    # ── Ucitaj podatke ────────────────────────────────────────
    sat  = read_as_rgb(sat_path)
    gt   = read_mask(mask_path)
    unet = read_mask(unet_path)
    dlv3 = read_mask(dlv3_path)

    # ── Kreiraj slike ─────────────────────────────────────────
    gt_ov   = mask_overlay(sat, gt)
    unet_ov = mask_overlay(sat, unet)
    dlv3_ov = mask_overlay(sat, dlv3)

    unet_diff = diff_overlay(sat, gt, unet)
    dlv3_diff = diff_overlay(sat, gt, dlv3)

    # ── Statistike za naslov ──────────────────────────────────
    pct_gt   = postoci_park(gt)
    pct_unet = postoci_park(unet)
    pct_dlv3 = postoci_park(dlv3)

    unet_err = (unet != gt).sum() / gt.size * 100
    dlv3_err = (dlv3 != gt).sum() / gt.size * 100

    # ── Figure: 2 reda × 3 kolone ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Primer {i+1:02d} — {stem}   |   Park u GT: {pct_gt:.1f}%",
        fontsize=13, weight='bold', y=0.98
    )

    # Red 1 — overlay predikcija
    axes[0, 0].imshow(gt_ov)
    axes[0, 0].set_title("Satelit + prava maska", fontsize=10, weight='bold')

    axes[0, 1].imshow(unet_ov)
    axes[0, 1].set_title(f"U-Net predikcija  ({pct_unet:.1f}% park)", fontsize=10)

    axes[0, 2].imshow(dlv3_ov)
    axes[0, 2].set_title(f"DeepLabV3+ predikcija  ({pct_dlv3:.1f}% park)", fontsize=10)

    # Red 2 — razlike u crvenoj boji
    axes[1, 0].imshow(sat)
    axes[1, 0].set_title("Satelit (referenca)", fontsize=10)

    axes[1, 1].imshow(unet_diff)
    axes[1, 1].set_title(
        f"U-Net razlika  ({unet_err:.1f}% pogresnih px)", fontsize=10
    )

    axes[1, 2].imshow(dlv3_diff)
    axes[1, 2].set_title(
        f"DeepLabV3+ razlika  ({dlv3_err:.1f}% pogresnih px)", fontsize=10
    )

    for ax in axes.flat:
        ax.axis('off')

    # Legenda
    handles = [
        mpatches.Patch(color='black',  label='Non-park  (crni overlay 67%)'),
        mpatches.Patch(color=(1, 0, 0, 0.6), label='Pogresna predikcija  (crveni overlay 60%)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_path = OUTPUT_DIR / f"primer_{i+1:02d}_{stem}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

    sacuvano += 1
    print(f"  [{sacuvano:2d}/20] {out_path.name}")

print(f"\nGotovo! Sacuvano {sacuvano} slika u: {OUTPUT_DIR.resolve()}")
