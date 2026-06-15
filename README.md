# Segmentacija parkova na satelitskim snimcima

Projekat iz predmeta **Inteligentni Sistemi** — duboka neuronska mreža koja automatski prepoznaje parkove na satelitskim snimcima gradova i generiše masku zelenih površina.

![Status](https://img.shields.io/badge/status-radi_se-yellow)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

---

## Sadržaj

- [O projektu](#o-projektu)
- [Dataset](#dataset)
- [Arhitektura modela](#arhitektura-modela)
- [Rezultati](#rezultati)
- [Struktura projekta](#struktura-projekta)
- [Instalacija](#instalacija)
- [Korišćenje](#korišćenje)
- [Tim](#tim)

---

# O projektu

Cilj projekta je razvoj modela **semantičke segmentacije** koji za zadatu satelitsku sliku grada predviđa koji pikseli pripadaju parkovima. Model prima RGB satelitsku sliku i na izlazu daje binarnu masku (1 = park, 0 = nije park).

Praktična primena uključuje praćenje zelenih površina u urbanim sredinama, prostorno planiranje i analizu promena vegetacije kroz vreme.

Implementirane su i poređene **dve arhitekture** — U-Net i DeepLabV3+.



# Dataset

Dataset je **ručno kreiran** korišćenjem QGIS alata. Satelitski snimci i maske parkova prikupljeni su za **10 evropskih gradova**:

| Grad | Broj tile-ova |
|------|:---:|
| Berlin | 448 |
| Rim | 369 |
| Lajpcig | 342 |
| Hamburg | 321 |
| Keln | 295 |
| Dizeldorf | 240 |
| Dortmund | 229 |
| Stutgard | 192 |
| Hanover | 141 |
| **Ukupno** | **2578** |

Svaki tile je dimenzija **256×256 piksela**. Frankfurt je izdvojen kao **test grad** (nije korišćen u treningu).

## Problem koordinatnih sistema (CRS)

Tokom pripreme podataka identifikovan je ključni problem — **neusklađenost koordinatnih sistema** između satelitskih snimaka i maski:

| | Satelitski snimak | Maska (pre korekcije) |
|---|---|---|
| **CRS** | EPSG:3857 (Pseudo-Mercator) | EPSG:4326 (WGS84) |
| **Jedinice** | Metri | Stepeni |

Iako su obe slike imale iste dimenzije u pikselima, različiti georeferentni sistemi prouzrokovali su **prostorni pomeraj od 300–700 metara**. Problem je rešen **reprojekcijom maski na EPSG:3857** uz nearest-neighbor resampling (`src/data_processing/warp_maski.py`).
## Dataset i modeli

 Dataset i istrenirani modeli nisu uključeni u repozitorijum zbog veličine.
 
 Možete ih preuzeti sa sledećeg linka: [Preuzmi dataset i modele](https://drive.google.com/drive/folders/1pUWCpfJck2wbj9SwTaonoepmB9jboMqk?usp=drive_link) 
 
 Nakon preuzimanja, raspakujte sadržaj u sledeću strukturu:
 
```
is_park_segmentation/
├── podaci/
│   ├── dataset_novi/        # dataset pre nego sto standardizujemo imena tile-ova
│   ├── provera_novi/        # rucna provera dataset-a
│   ├── raw/                 # tif fajlovi maski pre warpinga
│   ├── raw_novi/            # sateltiski snimci i maske, u tif formatu, pre tileing-a
│   ├── test/                # folder sa test podacima, i predikcijama modela
│   └── tiles_novi/          # ovo je actually dataset
```
 


<br>

#  Arhitektura modela

## U-Net

Glavna arhitektura — enkoder-dekoder mreža sa **skip connections** koje čuvaju prostorne informacije za precizne granice.

- **Encoder:** ResNet34 (pretreniran na ImageNet — transfer learning)
- **Decoder:** rekonstruiše masku uz pomoć skip connection-a sa svakog nivoa enkodera

## DeepLabV3+

Alternativna arhitektura sa **atrous (dilated) konvolucijama** i ASPP modulom koji hvata kontekst na više skala istovremeno.

## Parametri treninga

| Parametar | Vrednost |
|-----------|----------|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Dice Loss |
| Scheduler | ReduceLROnPlateau |
| Batch size | 8 |
| Epohe | 50 |
| Augmentacija | Flip, Rotate90, BrightnessContrast |
| Hardware | NVIDIA RTX 3060 Ti |

---

# Rezultati

Evaluacija na **Frankfurtu** (test grad, 289 tile-ova, ~18.9M piksela):

| Metrika | U-Net |
|---------|:---:|
| Accuracy | 91.97% |
| Precision | 67.50% |
| Recall | 68.82% |
| F1 Score | 68.15% |
| IoU | 51.69% |

> **Visoka preciznost je delom posledica klasne neravnoteže (parkovi zauzimaju ~12% piksela). Zato su **F1 i IoU** relevantnije metrike za ocenu kvaliteta segmentacije.**

Poređenje oba modela sa grafikama dostupno je pokretanjem `src/uporedi_modele.py`.

---

## Struktura projekta

```
is_park_segmentation/
├── src/
│   ├── train.py                 # trening U-Net modela
│   ├── train_deeplabv3.py       # trening DeepLabV3+ modela
│   ├── predict.py               # univerzalna predikcija (oba modela)
│   ├── provera_pred.py          # evaluacija i metrike
│   ├── uporedi_modele.py        # poređenje modela + grafici
│   ├── vizualizuj_unet.py       # vizualizacija encoder/decoder mapa
│   └── data_processing/
│       ├── warp_maski.py        # reprojekcija maski (CRS fix)
│       ├── tile_city.py         # isecanje slika na tile-ove
│       └── build_tiles.py       # kreiranje uniformnog dataseta
├── podaci/                      # dataset (preuzeti zasebno)
├── models/                      # istrenirani modeli (preuzeti zasebno)
├── dokumentacija_projekta.docx  # detaljna dokumentacija
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Instalacija

> **Napomena:** Potreban je **Python 3.12** (PyTorch još ne podržava 3.13+).

```bash
# 1. Kreiraj virtuelno okruženje
py -3.12 -m venv venv312
venv312\Scripts\activate          # Windows
# source venv312/bin/activate     # Linux/Mac

# 2. Instaliraj PyTorch sa CUDA podrškom
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Instaliraj ostale zavisnosti
pip install segmentation-models-pytorch albumentations rasterio pandas matplotlib tqdm
```

Provera CUDA podrške:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Korišćenje

### Trening

```bash
python src/train.py                # U-Net
python src/train_deeplabv3.py      # DeepLabV3+
```

### Predikcija

```bash
python src/predict.py --arhitektura unet
python src/predict.py --arhitektura deeplabv3
```

### Evaluacija

```bash
python src/provera_pred.py --arhitektura unet
```

### Poređenje modela (grafici)

```bash
python src/uporedi_modele.py
```

### Vizualizacija rada mreže

```bash
python src/vizualizuj_unet.py
```

---

## Tim

- _Ime Prezime_ — _broj indeksa_
- _Ime Prezime_ — _broj indeksa_

**Predmet:** Inteligentni Sistemi
**Fakultet:** Fakultet tehničkih nauka, Univerzitet u Novom Sadu
**Godina:** 2025/2026

---

## Licenca

Projekat je razvijen u edukativne svrhe u okviru fakultetskog kursa.