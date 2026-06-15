"""
========================================================
  Tile-ovanje satelitskog snimka i maske
========================================================

Struktura izlaza:
  dataset/
  ├── satellite/
  │   ├── hamburg_0000.tif
  │   ├── hamburg_0001.tif
  │   └── ...
  └── mask/
      ├── hamburg_0000.tif
      ├── hamburg_0001.tif
      └── ...

Balansiranje:
  - Pozitivni tile-ovi: sadrže parkove (mask > MIN_PARK_RATIO)
  - Negativni tile-ovi: bez parkova, čuva se max MAX_NEGATIVE_RATIO × broj pozitivnih

Instalacija:mis
    pip install rasterio numpy tqdm
"""

import random
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
import os


# Putanja do foldera u kom se nalazi ova skripta (data_processing/)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CITY_NAME       = "frankfurt"               # ime grada → prefiks tile-ova
SATELLITE_PATH  = os.path.join(PROJECT_ROOT, "podaci", "raw_novi", "frankfurt.tiff")        # putanja do satelitskog snimka
MASK_PATH       = os.path.join(PROJECT_ROOT, "podaci", "raw_novi", "frankfurt_maska.tif")   # putanja do maske

TILE_SIZE       = 256                     # veličina tile-a u pikselima
OVERLAP         = 0                       # overlap između tile-ova (0 = bez overlapa)
MIN_PARK_RATIO  = 0.01                    # minimalno % parkova da se smatra pozitivnim
MAX_NEGATIVE_RATIO = 0.5                  # max negativnih = ovaj broj × broj pozitivnih

SEED            = 42                      # reproduktibilnost
OUTPUT_DIR      = Path(os.path.join(PROJECT_ROOT, "podaci", "predikcija"))

# ─────────────────────────────────────────────────────────

SAT_DIR  = OUTPUT_DIR / "satellite"
MASK_DIR = OUTPUT_DIR / "mask"
SAT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

STEP = TILE_SIZE - OVERLAP
random.seed(SEED)


def save_tile(sat_tile, mask_tile, window, sat_src, sat_profile, mask_profile, tile_name):
    """Čuva jedan par (satelit, maska) tile-ova."""
    sat_tile_profile  = sat_profile.copy()
    mask_tile_profile = mask_profile.copy()

    transform = rasterio.windows.transform(window, sat_src.transform)
    sat_tile_profile["transform"]  = transform
    mask_tile_profile["transform"] = transform

    with rasterio.open(SAT_DIR / tile_name, "w", **sat_tile_profile) as dst:
        dst.write(sat_tile)

    with rasterio.open(MASK_DIR / tile_name, "w", **mask_tile_profile) as dst:
        dst.write(mask_tile[np.newaxis, ...])


def tile_city():
    sat_path  = Path(SATELLITE_PATH)
    mask_path = Path(MASK_PATH)

    if not sat_path.exists():
        raise FileNotFoundError(f"Snimak nije pronađen: {sat_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Maska nije pronađena: {mask_path}")

    with rasterio.open(sat_path) as sat_src, rasterio.open(mask_path) as mask_src:

        W = sat_src.width
        H = sat_src.height

        assert sat_src.width  == mask_src.width,  "Širina snimka i maske se ne poklapaju!"
        assert sat_src.height == mask_src.height, "Visina snimka i maske se ne poklapaju!"

        print(f"  Snimak:    {W} × {H} px")
        print(f"  Tile size: {TILE_SIZE} × {TILE_SIZE} px")
        print(f"  Overlap:   {OVERLAP} px")

        # Grid tile-ova
        col_offsets = list(range(0, W - TILE_SIZE + 1, STEP))
        row_offsets = list(range(0, H - TILE_SIZE + 1, STEP))

        if col_offsets and col_offsets[-1] + TILE_SIZE < W:
            col_offsets.append(W - TILE_SIZE)
        if row_offsets and row_offsets[-1] + TILE_SIZE < H:
            row_offsets.append(H - TILE_SIZE)

        total = len(row_offsets) * len(col_offsets)
        print(f"  Grid:      {len(row_offsets)} × {len(col_offsets)} = {total} tile-ova")
        print()

        sat_profile  = sat_src.meta.copy()
        mask_profile = mask_src.meta.copy()
        sat_profile.update({"width": TILE_SIZE, "height": TILE_SIZE})
        mask_profile.update({"width": TILE_SIZE, "height": TILE_SIZE})

        # ── Prolaz 1: prikupi sve tile-ove, odvoji pozitivne i negativne ──
        positives = []  # (window, sat_tile, mask_tile)
        negatives = []

        pbar = tqdm(total=total, desc="Skeniranje")
        for row in row_offsets:
            for col in col_offsets:
                window    = Window(col, row, TILE_SIZE, TILE_SIZE)
                sat_tile  = sat_src.read(window=window)
                mask_tile = mask_src.read(1, window=window)

                # Preskoči potpuno prazne tile-ove (nodata)
                if sat_tile.max() == 0:
                    pbar.update(1)
                    continue

                if mask_tile.mean() >= MIN_PARK_RATIO:
                    positives.append((window, sat_tile, mask_tile))
                else:
                    negatives.append((window, sat_tile, mask_tile))

                pbar.update(1)
        pbar.close()

        # ── Balansiranje negativnih tile-ova ──
        max_negatives = int(len(positives) * MAX_NEGATIVE_RATIO)
        if len(negatives) > max_negatives:
            negatives = random.sample(negatives, max_negatives)

        all_tiles = positives + negatives
        random.shuffle(all_tiles)

        print(f"\n  Pozitivni (sa parkom):  {len(positives)}")
        print(f"  Negativni (bez parka):  {len(negatives)}")
        print(f"  Ukupno za čuvanje:      {len(all_tiles)}")
        print()

        # ── Prolaz 2: čuvanje ──
        pbar = tqdm(total=len(all_tiles), desc="Čuvanje")
        for idx, (window, sat_tile, mask_tile) in enumerate(all_tiles):
            tile_name = f"{CITY_NAME}_{idx:04d}.tif"
            save_tile(sat_tile, mask_tile, window, sat_src,
                      sat_profile, mask_profile, tile_name)
            pbar.update(1)
        pbar.close()

    print()
    print("=" * 45)
    print(f"  Grad:        {CITY_NAME}")
    print(f"  Pozitivni:   {len(positives)} tile-ova")
    print(f"  Negativni:   {len(negatives)} tile-ova")
    print(f"  Ukupno:      {len(all_tiles)} tile-ova")
    print(f"  Izlaz:       {OUTPUT_DIR.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    print("=" * 45)
    print(f"  Tile-ovanje → {CITY_NAME}")
    print("=" * 45)
    tile_city()