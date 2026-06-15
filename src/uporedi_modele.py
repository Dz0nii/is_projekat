"""
========================================================
  Poredjenje U-Net i DeepLabV3+ modela
========================================================

Pokrece predikciju za oba modela, racuna metrike i
pravi grafike poredjenja.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Uvozi funkcije iz drugih skripti
from testiranje import predict_tiles
from provera_pred import izracunaj_metrike

# ── Putanje ──────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PREDIKCIJA_PATH = os.path.join(PROJECT_ROOT, "podaci", "test")
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "poredjenje")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Definicija modela ────────────────────────────────────────
MODELI = {
    'U-Net': {
        'arhitektura': 'unet',
        'model_path':  os.path.join(MODELS_DIR, "Unet_model.pth"),
        'subfolder':   'prediktovana_maska_unet',
        'boja':        '#2E7D32',
    },
    'DeepLabV3+': {
        'arhitektura': 'deeplabv3',
        'model_path':  os.path.join(MODELS_DIR, "park_model_deeplabv3_best.pth"),
        'subfolder':   'prediktovana_maska_deeplabv3',
        'boja':        '#1565C0',
    },
}

# ── Korak 1: Pokreni predikciju za oba modela ────────────────
print("\n" + "#" * 50)
print("# KORAK 1: PREDIKCIJA")
print("#" * 50)

for naziv, info in MODELI.items():
    print(f"\n--- {naziv} ---")
    predict_tiles(
        PREDIKCIJA_PATH,
        model_path=info['model_path'],
        arhitektura=info['arhitektura'],
        output_subfolder=info['subfolder']
    )

# ── Korak 2: Izracunaj metrike za oba modela ─────────────────
print("\n" + "#" * 50)
print("# KORAK 2: METRIKE")
print("#" * 50)

rezultati = {}
for naziv, info in MODELI.items():
    m = izracunaj_metrike(PREDIKCIJA_PATH, info['subfolder'])
    rezultati[naziv] = m
    print(f"\n{naziv}:")
    print(f"  Accuracy:  {m['accuracy']*100:.2f}%")
    print(f"  Precision: {m['precision']*100:.2f}%")
    print(f"  Recall:    {m['recall']*100:.2f}%")
    print(f"  F1:        {m['f1']*100:.2f}%")
    print(f"  IoU:       {m['iou']*100:.2f}%")

# ── Korak 3: Grafik 1 - poredjenje metrika (bar chart) ───────
print("\n" + "#" * 50)
print("# KORAK 3: GRAFICI")
print("#" * 50)

metrike_nazivi = ['Accuracy', 'Precision', 'Recall', 'F1', 'IoU']
metrike_kljucevi = ['accuracy', 'precision', 'recall', 'f1', 'iou']

x = np.arange(len(metrike_nazivi))
sirina = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
for i, (naziv, info) in enumerate(MODELI.items()):
    vrednosti = [rezultati[naziv][k] * 100 for k in metrike_kljucevi]
    offset = (i - 0.5) * sirina
    bars = ax.bar(x + offset, vrednosti, sirina, label=naziv, color=info['boja'])
    # Vrednosti iznad stubica
    for bar, v in zip(bars, vrednosti):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Procenat (%)')
ax.set_title('Poredjenje metrika: U-Net vs DeepLabV3+')
ax.set_xticks(x)
ax.set_xticklabels(metrike_nazivi)
ax.legend()
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_poredjenje_metrika.png"), dpi=120)
plt.close()
print("Sacuvano: 1_poredjenje_metrika.png")

# ── Grafik 2 - confusion matrix komponente ───────────────────
fig, axes = plt.subplots(1, len(MODELI), figsize=(6*len(MODELI), 5))
if len(MODELI) == 1:
    axes = [axes]

for ax, (naziv, info) in zip(axes, MODELI.items()):
    m = rezultati[naziv]
    matrica = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    im = ax.imshow(matrica, cmap='Blues')
    ax.set_title(f'{naziv} - Confusion Matrix')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred: Ne-park', 'Pred: Park'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Stvarno: Ne-park', 'Stvarno: Park'])
    # Brojevi u celijama
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{matrica[r,c]:,}', ha='center', va='center',
                    color='white' if matrica[r,c] > matrica.max()/2 else 'black',
                    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix.png"), dpi=120)
plt.close()
print("Sacuvano: 2_confusion_matrix.png")

# ── Grafik 3 - tabela rezultata ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3))
ax.axis('off')

kolone = ['Model'] + metrike_nazivi
podaci = []
for naziv in MODELI:
    red = [naziv] + [f"{rezultati[naziv][k]*100:.2f}%" for k in metrike_kljucevi]
    podaci.append(red)

tabela = ax.table(cellText=podaci, colLabels=kolone, loc='center', cellLoc='center')
tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1, 2)
# Boja header reda
for j in range(len(kolone)):
    tabela[0, j].set_facecolor('#2E7D32')
    tabela[0, j].set_text_props(color='white', weight='bold')

ax.set_title('Tabela rezultata', pad=20, fontsize=13, weight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "3_tabela_rezultata.png"), dpi=120, bbox_inches='tight')
plt.close()
print("Sacuvano: 3_tabela_rezultata.png")

# ── Zakljucak ────────────────────────────────────────────────
print("\n" + "#" * 50)
print("# ZAKLJUCAK")
print("#" * 50)
najbolji_f1  = max(MODELI, key=lambda n: rezultati[n]['f1'])
najbolji_iou = max(MODELI, key=lambda n: rezultati[n]['iou'])
print(f"Najbolji F1:  {najbolji_f1} ({rezultati[najbolji_f1]['f1']*100:.2f}%)")
print(f"Najbolji IoU: {najbolji_iou} ({rezultati[najbolji_iou]['iou']*100:.2f}%)")
print(f"\nSvi grafici sacuvani u: {OUTPUT_DIR}")