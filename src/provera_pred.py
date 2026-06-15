import os
import numpy as np
import rasterio
from pathlib import Path

def izracunaj_metrike(predikcija_folder, prediktovana_subfolder):
    """Poredi tacnu masku sa predvidjenom i vraca recnik metrika."""
    tacna_folder        = Path(predikcija_folder) / 'mask'
    prediktovana_folder = Path(predikcija_folder) / prediktovana_subfolder

    tiles = sorted(tacna_folder.glob('*.tif'))

    ukupno_piksela = 0
    tp = tn = fp = fn = 0

    for tile_path in tiles:
        pred_path = prediktovana_folder / tile_path.name
        if not pred_path.exists():
            continue

        with rasterio.open(tile_path) as src:
            tacna = (src.read(1) > 0).astype(np.uint8)
        with rasterio.open(pred_path) as src:
            pred = (src.read(1) > 0).astype(np.uint8)

        tp += np.sum((pred == 1) & (tacna == 1))
        tn += np.sum((pred == 0) & (tacna == 0))
        fp += np.sum((pred == 1) & (tacna == 0))
        fn += np.sum((pred == 0) & (tacna == 1))
        ukupno_piksela += tacna.size

    tacnost    = (tp + tn) / ukupno_piksela if ukupno_piksela > 0 else 0
    preciznost = tp / (tp + fp) if (tp + fp) > 0 else 0
    odziv      = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1         = 2 * preciznost * odziv / (preciznost + odziv) if (preciznost + odziv) > 0 else 0
    iou        = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'ukupno_piksela': int(ukupno_piksela),
        'accuracy': tacnost, 'precision': preciznost,
        'recall': odziv, 'f1': f1, 'iou': iou
    }


def proveri_poklapanje(predikcija_folder, prediktovana_subfolder, naziv=""):
    """Ispisuje metrike za jedan model."""
    m = izracunaj_metrike(predikcija_folder, prediktovana_subfolder)

    print("=" * 45)
    print(f"REZULTATI POKLAPANJA {naziv}")
    print("=" * 45)
    print(f"Ukupno piksela:      {m['ukupno_piksela']:,}")
    print(f"Tacno park (TP):     {m['tp']:,}")
    print(f"Tacno ne-park (TN):  {m['tn']:,}")
    print(f"Pogresno park (FP):  {m['fp']:,}")
    print(f"Propusten park (FN): {m['fn']:,}")
    print()
    print(f"Tacnost (Accuracy):     {m['accuracy']*100:.2f}%")
    print(f"Preciznost (Precision): {m['precision']*100:.2f}%")
    print(f"Odziv (Recall):         {m['recall']*100:.2f}%")
    print(f"F1 Score:               {m['f1']*100:.2f}%")
    print(f"IoU:                    {m['iou']*100:.2f}%")
    print("=" * 45)
    return m


if __name__ == '__main__':
    import argparse
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--arhitektura', default='unet', choices=['unet', 'deeplabv3'])
    args = parser.parse_args()

    predikcija_path = os.path.join(PROJECT_ROOT, "podaci", "test")
    subfolder = f'prediktovana_maska_{args.arhitektura}'
    proveri_poklapanje(predikcija_path, subfolder, naziv=f"({args.arhitektura})")