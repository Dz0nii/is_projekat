"""
=============================================================
  Evolucija treninga — metrike i vizualizacija kroz epohe
=============================================================

1. Ucitava sve checkpoint modele iz models/evolucija/unet/ i deeplabv3/
2. Pokrece predikciju za svaki checkpoint na test skupu
3. Racuna metrike (F1, IoU, Accuracy...) za svaki checkpoint
4. Generise:
     - poredjenje/evolucija/kriva_ucenja.png
         Kriva ucenja: F1 i IoU kroz epohe za oba modela
     - vizualizacija_poredjenja/evolucija/napredak_TILE.png
         Za N tile-ova: redovi = epohe, kolone = GT | UNet | DLV3 | UNet diff | DLV3 diff

Pokretanje:
  python src/evolucija_treninga.py
"""

import os
import re
import random
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from testiranje import predict_tiles
from provera_pred import izracunaj_metrike

# ── Putanje ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_DIR      = Path(PROJECT_ROOT) / "podaci" / "test"
EVOLUCIJA_DIR = Path(PROJECT_ROOT) / "models" / "evolucija"
OUTPUT_METR   = Path(PROJECT_ROOT) / "poredjenje" / "evolucija"
OUTPUT_VIZ    = Path(PROJECT_ROOT) / "vizualizacija_poredjenja" / "evolucija"
OUTPUT_METR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIZ.mkdir(parents=True, exist_ok=True)

N_TILE_VIZ = 5   # koliko tile-ova prikazati u vizualizaciji napretka
SEED        = 42

MODELI_INFO = {
    'unet': {
        'naziv': 'U-Net',
        'boja':  '#2E7D32',
        'folder': EVOLUCIJA_DIR / 'unet',
    },
    'deeplabv3': {
        'naziv': 'DeepLabV3+',
        'boja':  '#1565C0',
        'folder': EVOLUCIJA_DIR / 'deeplabv3',
    },
}


# ── Pomocne funkcije ──────────────────────────────────────────────────────────

def parse_epoch(path: Path):
    """Sortirni kljuc: (broj_epohe, tag). Best dobija 999 da bude poslednji."""
    name = path.stem
    if 'best' in name:
        return (999, 'best')
    m = re.search(r'epoch_(\d+)', name)
    if m:
        return (int(m.group(1)), None)
    return (0, None)


def read_as_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read()
    if data.shape[0] == 1:
        band = data[0].astype(np.float32)
        band = (band - band.min()) / (band.max() - band.min() + 1e-8)
        return np.stack([band, band, band], axis=-1)
    rgb = data[:3].astype(np.float32).transpose(1, 2, 0)
    for c in range(3):
        ch = rgb[:, :, c]
        rgb[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return rgb


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(np.uint8)


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, opacity: float = 0.67) -> np.ndarray:
    result = rgb.copy()
    result[mask == 0] *= (1.0 - opacity)
    return np.clip(result, 0, 1)


def diff_overlay(rgb: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray,
                 opacity: float = 0.60) -> np.ndarray:
    result = rgb.copy()
    diff = pred_mask != true_mask
    result[diff] = result[diff] * (1.0 - opacity) + np.array([1., 0., 0.]) * opacity
    return np.clip(result, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# KORAK 1 — Predikcija i metrike za sve checkpointe
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KORAK 1: Predikcija i metrike za sve checkpointe")
print("=" * 60)

# rezultati[arhitektura] = lista diktova {epoch, label, subfolder, metrike}
rezultati = {k: [] for k in MODELI_INFO}

for arhitektura, info in MODELI_INFO.items():
    folder = info['folder']
    if not folder.exists():
        print(f"\n[!] Nema foldera: {folder}")
        print(f"    Pokreni train_{arhitektura.replace('deeplab', 'dlv').replace('v3', 'three')}.py prvo.")
        continue

    checkpointi = sorted(folder.glob("*.pth"), key=parse_epoch)
    if not checkpointi:
        print(f"\n[!] Nema .pth fajlova u: {folder}")
        continue

    print(f"\n{info['naziv']}: {len(checkpointi)} checkpoint(a) pronadjeno")

    for ckpt in checkpointi:
        ep_num, ep_tag = parse_epoch(ckpt)
        label     = "best" if ep_tag == 'best' else f"epoch {ep_num}"
        subfolder = f"evolucija_{arhitektura}_{ckpt.stem}"

        print(f"  → {ckpt.name:<35} ({label})", end="", flush=True)

        predict_tiles(
            str(TEST_DIR),
            model_path=str(ckpt),
            arhitektura=arhitektura,
            output_subfolder=subfolder,
        )

        m = izracunaj_metrike(str(TEST_DIR), subfolder)
        rezultati[arhitektura].append({
            'epoch':     ep_num,
            'label':     label,
            'subfolder': subfolder,
            'metrike':   m,
        })

        print(f"  F1={m['f1']*100:.1f}%  IoU={m['iou']*100:.1f}%  Acc={m['accuracy']*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# KORAK 2 — Kriva ucenja (F1 i IoU kroz epohe)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KORAK 2: Kriva ucenja")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Evolucija treninga — napredak kroz epohe", fontsize=13, weight='bold')

METRIKE_VIZ = [
    ('f1',  'F1 Score (%)'),
    ('iou', 'IoU (%)'),
]

for ax, (kljuc, naziv) in zip(axes, METRIKE_VIZ):
    for arhitektura, info in MODELI_INFO.items():
        stavke = rezultati[arhitektura]
        if not stavke:
            continue

        # Tacke za snapshot epohe (bez best)
        snapshot_stavke = [s for s in stavke if s['label'] != 'best']
        if snapshot_stavke:
            x = [s['epoch'] for s in snapshot_stavke]
            y = [s['metrike'][kljuc] * 100 for s in snapshot_stavke]
            ax.plot(x, y, 'o-', color=info['boja'], label=info['naziv'],
                    linewidth=2, markersize=6, zorder=3)

        # Horizontalna isprekidana linija za best model
        best = next((s for s in stavke if s['label'] == 'best'), None)
        if best:
            val = best['metrike'][kljuc] * 100
            ax.axhline(val, color=info['boja'], linestyle='--', alpha=0.55, linewidth=1.5,
                       label=f"{info['naziv']} best ({val:.1f}%)")

    ax.set_xlabel("Epoha", fontsize=10)
    ax.set_ylabel(naziv, fontsize=10)
    ax.set_title(naziv, fontsize=11, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

plt.tight_layout()
out_kriva = OUTPUT_METR / "kriva_ucenja.png"
plt.savefig(out_kriva, dpi=130, bbox_inches='tight')
plt.close()
print(f"Sacuvano: {out_kriva}")

# ── Tabela metrika po epohama ─────────────────────────────────────────────────

sve_epohe = sorted(set(
    s['epoch'] for arh in MODELI_INFO for s in rezultati[arh]
))

for arhitektura, info in MODELI_INFO.items():
    stavke = rezultati[arhitektura]
    if not stavke:
        continue

    ep_map = {s['epoch']: s for s in stavke}
    epohe_za_prikaz = [e for e in sve_epohe if e in ep_map]

    n = len(epohe_za_prikaz)
    if n == 0:
        continue

    labele  = [ep_map[e]['label'] for e in epohe_za_prikaz]
    acc_v   = [ep_map[e]['metrike']['accuracy']  * 100 for e in epohe_za_prikaz]
    prec_v  = [ep_map[e]['metrike']['precision'] * 100 for e in epohe_za_prikaz]
    rec_v   = [ep_map[e]['metrike']['recall']    * 100 for e in epohe_za_prikaz]
    f1_v    = [ep_map[e]['metrike']['f1']        * 100 for e in epohe_za_prikaz]
    iou_v   = [ep_map[e]['metrike']['iou']       * 100 for e in epohe_za_prikaz]

    podaci_tabele = [
        [lbl, f"{a:.1f}%", f"{p:.1f}%", f"{r:.1f}%", f"{f:.1f}%", f"{i:.1f}%"]
        for lbl, a, p, r, f, i in zip(labele, acc_v, prec_v, rec_v, f1_v, iou_v)
    ]
    kolone_tabele = ['Epoha', 'Accuracy', 'Precision', 'Recall', 'F1', 'IoU']

    fig_t, ax_t = plt.subplots(figsize=(10, 0.5 * n + 1.5))
    ax_t.axis('off')
    tabela = ax_t.table(
        cellText=podaci_tabele, colLabels=kolone_tabele,
        loc='center', cellLoc='center'
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1, 1.6)
    for j in range(len(kolone_tabele)):
        tabela[0, j].set_facecolor(info['boja'])
        tabela[0, j].set_text_props(color='white', weight='bold')

    ax_t.set_title(f"Metrike kroz epohe — {info['naziv']}", fontsize=11, weight='bold', pad=10)
    out_t = OUTPUT_METR / f"tabela_epohe_{arhitektura}.png"
    plt.savefig(out_t, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Sacuvano: {out_t}")


# ══════════════════════════════════════════════════════════════════════════════
# KORAK 3 — Vizualizacija napretka predikcija kroz epohe (po tile-u)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KORAK 3: Vizualizacija napretka predikcija")
print("=" * 60)

sat_dir  = TEST_DIR / "satellite"
mask_dir = TEST_DIR / "mask"

# Spoji epohe koje postoje u OBA modela (inner join po broju epohe)
ep_unet = {s['epoch']: s for s in rezultati['unet']}
ep_dlv3 = {s['epoch']: s for s in rezultati['deeplabv3']}
zajednicke_epohe = sorted(ep_unet.keys() & ep_dlv3.keys())
redovi = [(ep_unet[e], ep_dlv3[e]) for e in zajednicke_epohe]

if not redovi:
    print("[!] Nema zajednickih epoha izmedju modela — preskacam vizualizaciju.")
else:
    # Odaberi tile-ove za koje postoje sve predikcije
    def tile_kompletan(stem: str) -> bool:
        for st_u, st_d in redovi:
            if not (TEST_DIR / st_u['subfolder'] / f"{stem}.tif").exists():
                return False
            if not (TEST_DIR / st_d['subfolder'] / f"{stem}.tif").exists():
                return False
        return True

    validni = [t for t in sorted(sat_dir.glob("*.tif")) if tile_kompletan(t.stem)]
    if not validni:
        print("[!] Nema tile-ova sa svim predikcijama — preskacam vizualizaciju.")
    else:
        random.seed(SEED)
        odabrani = random.sample(validni, min(N_TILE_VIZ, len(validni)))
        print(f"Vizualizujem {len(odabrani)} tile-ova x {len(redovi)} epoha...")

        N_KOL = 5   # GT | UNet | DLV3 | UNet diff | DLV3 diff
        KOL_NASLOVI = [
            "Satelit + GT maska",
            "U-Net predikcija",
            "DeepLabV3+ predikcija",
            "U-Net razlika",
            "DeepLabV3+ razlika",
        ]

        for tile_path in odabrani:
            stem = tile_path.stem
            sat  = read_as_rgb(tile_path)
            gt   = read_mask(mask_dir / f"{stem}.tif")
            gt_ov = mask_overlay(sat, gt)

            n_red = len(redovi)
            fig, axes = plt.subplots(
                n_red, N_KOL,
                figsize=(N_KOL * 3.0, n_red * 3.0),
                squeeze=False
            )

            fig.suptitle(
                f"Napredak predikcija kroz epohe — {stem}",
                fontsize=11, weight='bold', y=1.01
            )

            for j, naslov in enumerate(KOL_NASLOVI):
                axes[0, j].set_title(naslov, fontsize=8, weight='bold', pad=4)

            for i, (st_u, st_d) in enumerate(redovi):
                ep_label = st_u['label'].replace("epoch ", "Epoha ")
                if ep_label == 'best':
                    ep_label = 'Best'

                unet_pred = read_mask(TEST_DIR / st_u['subfolder'] / f"{stem}.tif")
                dlv3_pred = read_mask(TEST_DIR / st_d['subfolder'] / f"{stem}.tif")

                slike = [
                    gt_ov,
                    mask_overlay(sat, unet_pred),
                    mask_overlay(sat, dlv3_pred),
                    diff_overlay(sat, gt, unet_pred),
                    diff_overlay(sat, gt, dlv3_pred),
                ]

                for j, slika in enumerate(slike):
                    axes[i, j].imshow(slika)
                    axes[i, j].axis('off')

                f1_u = st_u['metrike']['f1'] * 100
                f1_d = st_d['metrike']['f1'] * 100
                axes[i, 0].set_ylabel(
                    f"{ep_label}\nU: F1={f1_u:.0f}%\nD: F1={f1_d:.0f}%",
                    fontsize=7, rotation=0, ha='right', va='center',
                    labelpad=70
                )

            handles = [
                mpatches.Patch(color='black',       label='Non-park (tamniji)'),
                mpatches.Patch(color=(1, 0, 0, 0.6), label='Pogresna predikcija (crveno)'),
            ]
            fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8,
                       framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

            plt.tight_layout()
            out_path = OUTPUT_VIZ / f"napredak_{stem}.png"
            plt.savefig(out_path, dpi=110, bbox_inches='tight')
            plt.close()
            print(f"  Sacuvano: {out_path.name}")

print("\n" + "=" * 60)
print("GOTOVO!")
print(f"  Kriva ucenja + tabele: {OUTPUT_METR.resolve()}")
print(f"  Vizualizacije napretka: {OUTPUT_VIZ.resolve()}")
print("=" * 60)
