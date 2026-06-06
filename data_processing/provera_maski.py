"""
========================================================
  Vizualizacija maski — provera kvaliteta anotacija
========================================================

Za svaki tile iz dataset/satellite/ i dataset/mask/
generiše sliku gde su:
  - Parkovi: originalna boja (čista)
  - Ostalo:  zatamljeno (multiply faktor)

Izlaz: provera/<city>_XXXX_provera.png

Instalacija:
    pip install rasterio numpy Pillow tqdm
"""

import numpy as np
import rasterio
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
#  PODEŠAVANJA — menjaj ovde
# ─────────────────────────────────────────────────────────

CITY_NAME   = "hamburg"       # mora da se poklapa sa tile_city.py
DARKEN      = 0.35            # koliko zatamniti non-park (0=crno, 1=originalno)

DATASET_DIR = Path("dataset")
OUTPUT_DIR  = Path("provera")

# ─────────────────────────────────────────────────────────

SAT_DIR  = DATASET_DIR / "satellite"
MASK_DIR = DATASET_DIR / "mask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_as_rgb(path: Path) -> np.ndarray:
    """
    Čita TIF i vraća RGB numpy array (H, W, 3) u uint8.
    Ako ima više od 3 kanala, uzima prva 3 (B, G, R → R, G, B).
    Ako ima 1 kanal (grayscale), konvertuje u RGB.
    """
    with rasterio.open(path) as src:
        data = src.read()  # (C, H, W)

    if data.shape[0] == 1:
        # Grayscale → RGB
        band = data[0].astype(np.float32)
        band = (band - band.min()) / (band.max() - band.min() + 1e-8)
        rgb = np.stack([band, band, band], axis=-1)
    else:
        # Uzmi prva 3 kanala
        rgb = data[:3].astype(np.float32)  # (3, H, W)
        rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

        # Normalizuj po kanalu na [0, 1]
        for c in range(3):
            ch = rgb[:, :, c]
            mn, mx = ch.min(), ch.max()
            rgb[:, :, c] = (ch - mn) / (mx - mn + 1e-8)

    return (rgb * 255).astype(np.uint8)


def read_mask(path: Path) -> np.ndarray:
    """Čita masku i vraća binarni (H, W) uint8 array."""
    with rasterio.open(path) as src:
        mask = src.read(1)
    return (mask > 0).astype(np.uint8)


def apply_overlay(rgb: np.ndarray, mask: np.ndarray, darken: float) -> np.ndarray:
    """
    Zatamljuje non-park piksele.
    Park pikseli ostaju originalni.
    """
    result = rgb.astype(np.float32)

    # Non-park maska
    non_park = (mask == 0)

    # Zatamljenje
    result[non_park] = result[non_park] * darken

    return result.clip(0, 255).astype(np.uint8)


def process_city():
    # Pronađi sve satelitske tile-ove za grad
    tiles = sorted(SAT_DIR.glob(f"{CITY_NAME}_*.tif"))

    if not tiles:
        print(f"  ❌ Nisu pronađeni tile-ovi za grad: {CITY_NAME}")
        print(f"     Proveri da li postoje fajlovi u: {SAT_DIR}")
        return

    print(f"  Pronađeno: {len(tiles)} tile-ova")
    print(f"  Darken faktor: {DARKEN}")
    print()

    saved    = 0
    skipped  = 0

    for sat_path in tqdm(tiles, desc=f"Generišem proveru"):
        tile_stem = sat_path.stem            # npr. "hamburg_0042"
        mask_path = MASK_DIR / f"{tile_stem}.tif"

        if not mask_path.exists():
            skipped += 1
            continue

        # Učitaj
        rgb  = read_as_rgb(sat_path)
        mask = read_mask(mask_path)

        # Proveri dimenzije
        if rgb.shape[:2] != mask.shape:
            skipped += 1
            continue

        # Primeni overlay
        result = apply_overlay(rgb, mask, DARKEN)

        # Sačuvaj kao PNG
        out_name = f"{tile_stem}_provera.png"
        Image.fromarray(result).save(OUTPUT_DIR / out_name)
        saved += 1

    print()
    print("=" * 45)
    print(f"  Grad:      {CITY_NAME}")
    print(f"  Sačuvano:  {saved} slika")
    print(f"  Preskočeno: {skipped}")
    print(f"  Izlaz:     {OUTPUT_DIR.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    print("=" * 45)
    print(f"  Provera maski — {CITY_NAME}")
    print("=" * 45)
    process_city()
