"""
========================================================
  Kreiranje uniformnog tiles/ foldera
========================================================

Uzima sve fajlove iz dataset/satellite/ i dataset/mask/,
uparuje ih po imenu i kopira u tiles/ folder sa uniformnim
imenima tile_0000.tif, tile_0001.tif, ...

Čuva i CSV mapu: tiles/tile_map.csv
  tile_name, original_name, city
  tile_0000.tif, hamburg_0042.tif, hamburg

Instalacija:
    pip install tqdm
"""

import shutil
import csv
from pathlib import Path
from tqdm import tqdm
import os

# ─────────────────────────────────────────────────────────
#  PODEŠAVANJA
# ─────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = Path(os.path.join(PROJECT_ROOT, "podaci", "dataset_novi"))
TILES_DIR   = Path(os.path.join(PROJECT_ROOT, "podaci", "tiles_novi"))


# ─────────────────────────────────────────────────────────

SAT_IN   = DATASET_DIR / "satellite"
MASK_IN  = DATASET_DIR / "mask"
SAT_OUT  = TILES_DIR / "satellite"
MASK_OUT = TILES_DIR / "mask"

SAT_OUT.mkdir(parents=True, exist_ok=True)
MASK_OUT.mkdir(parents=True, exist_ok=True)


def get_city(filename: str) -> str:
    """Izvlači ime grada iz naziva fajla. npr. 'hamburg_0042' → 'hamburg'"""
    parts = filename.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else "unknown"


def build_tiles():
    # Pronađi sve satelitske fajlove i sortiraj
    sat_files = sorted(SAT_IN.glob("*.tif"))

    if not sat_files:
        print(f"❌ Nema fajlova u {SAT_IN}")
        return

    # Proveri uparene maske
    paired   = []
    unpaired = []

    for sat_path in sat_files:
        mask_path = MASK_IN / sat_path.name
        if mask_path.exists():
            paired.append((sat_path, mask_path))
        else:
            unpaired.append(sat_path.name)

    if unpaired:
        print(f"⚠️  {len(unpaired)} fajlova nema par u mask/ — preskačem:")
        for name in unpaired[:5]:
            print(f"   {name}")
        if len(unpaired) > 5:
            print(f"   ... i još {len(unpaired) - 5}")

    print(f"\n  Uparenih tile-ova: {len(paired)}")
    print(f"  Gradovi: {sorted(set(get_city(p[0].stem) for p in paired))}")
    print()

    # Kopiraj sa novim imenima
    tile_map = []

    for idx, (sat_path, mask_path) in enumerate(tqdm(paired, desc="Kopiram")):
        tile_name = f"tile_{idx:04d}.tif"
        city      = get_city(sat_path.stem)

        shutil.copy2(sat_path,  SAT_OUT  / tile_name)
        shutil.copy2(mask_path, MASK_OUT / tile_name)

        tile_map.append({
            "tile_name":     tile_name,
            "original_name": sat_path.name,
            "city":          city,
        })

    # Sačuvaj CSV mapu
    csv_path = TILES_DIR / "tile_map.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_name", "original_name", "city"])
        writer.writeheader()
        writer.writerows(tile_map)

    # Statistike po gradu
    city_counts = {}
    for row in tile_map:
        city_counts[row["city"]] = city_counts.get(row["city"], 0) + 1

    print()
    print("=" * 45)
    print(f"  Ukupno tile-ova: {len(tile_map)}")
    print(f"  Po gradu:")
    for city, count in sorted(city_counts.items()):
        print(f"    {city:<20} {count:>5} tile-ova")
    print(f"  Mapa sačuvana:   {csv_path}")
    print(f"  Izlaz:           {TILES_DIR.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    print("=" * 45)
    print("  Kreiranje uniformnog tiles/ foldera")
    print("=" * 45)
    build_tiles()