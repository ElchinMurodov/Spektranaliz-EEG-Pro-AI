"""
dataset.py — ML uchun BELGILAR MATRITSASINI tayyorlash.

Har bir EEG yozuvini bitta sonli belgi-vektoriga aylantiradi. Vektor IKKI
guruh belgidan iborat:
  (1) STATIK spektral belgilar  — features.extract_features (nisbiy quvvatlar,
      nisbatlar, engagement, iAPF, FAA, FMT, entropiya, dominant, edge).
  (2) DINAMIK vaqt-chastota belgilar — timefreq.dynamic_features (har ritm
      quvvatining vaqt bo'yicha o'rtacha/o'zgaruvchanlik/trend/diapazon).

Belgi sxemasi qurilma/kanal soni/formatdan QAT'IY NAZAR bir xil bo'ladi
(kanallar o'rtachalanadi), shuning uchun CONTEC KT-88 (16 ch), Нейротех
"Компакт-нейро" (21 ch) va CSV yozuvlari bitta modelda birga ishlatiladi.

Yorliqlar (labels):
  * labels.csv orqali  — "file,label" ustunlari (haqiqiy sportchi ma'lumoti).
  * yoki fayl nomidan  — synth_<holat>_<fs>.* ko'rinishidagi sun'iy fayllardan.
"""

import os
import csv as _csv

from . import loader, preprocessing, spectral, features, timefreq, config


# Statik belgilar (features.extract_features dan; '_regions' tashqari, faa/fmt
# None bo'lishi mumkin -> 0.0 bilan to'ldiriladi).
STATIC_FEATURES = [
    "rp_delta", "rp_theta", "rp_alpha", "rp_beta", "rp_gamma",
    "ratio_alpha_beta", "ratio_theta_beta", "ratio_theta_alpha", "ratio_beta_alpha",
    "engagement", "iapf", "faa", "fmt",
    "spectral_entropy", "dominant_frequency", "fft_dominant_frequency",
    "spectral_edge_95",
]

# Dinamik belgilar (timefreq.dynamic_features dan)
DYNAMIC_FEATURES = []
for _b in config.BANDS:
    for _stat in ("mean", "std", "trend", "range"):
        DYNAMIC_FEATURES.append("%s_dyn_%s" % (_b, _stat))

FEATURE_NAMES = STATIC_FEATURES + DYNAMIC_FEATURES

# Sun'iy fayl nomidagi qisqartma -> rasmiy holat nomi
_SYNTH_LABELS = {
    "normal": "Normal",
    "fokus": "Fokus",
    "charchoq": "Charchoq",
    "uyquga_moyillik": "Uyquga moyillik",
    "uyquga": "Uyquga moyillik",
    "uyqusizlik": "Uyqusizlik",
    "qozgalish": "Qo'zg'alish / Xayojonlanish",
    "qo'zg'alish": "Qo'zg'alish / Xayojonlanish",
    "stress": "Stress",
    "meditativ": "Meditativ",
}


def feature_vector(rec, spec=None, include_dynamic=True):
    """Bitta Recording -> (FEATURE_NAMES tartibida) sonli vektor."""
    if spec is None:
        spec = spectral.analyze_recording(rec)
    feats = features.extract_features(rec, spec)
    vec = []
    for k in STATIC_FEATURES:
        v = feats.get(k)
        vec.append(float(v) if v is not None else 0.0)
    if include_dynamic:
        dyn = timefreq.dynamic_features(rec)
        for k in DYNAMIC_FEATURES:
            vec.append(float(dyn.get(k, 0.0)))
    names = FEATURE_NAMES if include_dynamic else STATIC_FEATURES
    return names, vec


def label_from_filename(path):
    """synth_<holat>_<fs>.* fayl nomidan holat yorlig'ini chiqaradi (bo'lmasa None)."""
    base = os.path.basename(path).lower()
    name = os.path.splitext(base)[0]
    if name.startswith("synth_"):
        name = name[len("synth_"):]
    # eng uzun mos kalitni tanlash (masalan "uyquga_moyillik")
    for key in sorted(_SYNTH_LABELS, key=len, reverse=True):
        if name.startswith(key):
            return _SYNTH_LABELS[key]
    return None


def load_labels_csv(path):
    """labels.csv ni o'qiydi: 'file,label' ustunlari. {basename: label} qaytaradi."""
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = _csv.reader(fh)
        rows = [r for r in reader if r and len(r) >= 2]
    start = 0
    if rows and rows[0][0].strip().lower() in ("file", "fayl", "filename"):
        start = 1
    for r in rows[start:]:
        out[os.path.basename(r[0].strip())] = r[1].strip()
    return out


def _list_eeg_files(path):
    if os.path.isfile(path):
        return [path]
    exts = (".edf", ".bdf", ".csv")
    files = []
    for root, _dirs, names in os.walk(path):
        for nm in sorted(names):
            if os.path.splitext(nm)[1].lower() in exts:
                files.append(os.path.join(root, nm))
    return files


def build_dataset(path, labels_csv=None, infer_from_name=False,
                  include_dynamic=True, fs=None, target_fs=None, notch=True,
                  verbose=True):
    """
    Papka (yoki fayl) -> (feature_names, X, y, files, skipped).

    Yorliq manbai:
      * labels_csv berilsa — o'sha fayldan (haqiqiy ma'lumot uchun).
      * infer_from_name=True — fayl nomidan (sun'iy fayllar uchun).
      * ikkalasi ham yo'q -> y = [None, ...] (klasterlash / faqat belgilar uchun).
    """
    files = _list_eeg_files(path)
    label_map = load_labels_csv(labels_csv) if labels_csv else {}

    names = FEATURE_NAMES if include_dynamic else STATIC_FEATURES
    X, y, used, skipped = [], [], [], []
    for f in files:
        try:
            rec = loader.load(f, fs=fs)
            preprocessing.preprocess(rec, target_fs=target_fs, notch=notch)
            spec = spectral.analyze_recording(rec)
            _n, vec = feature_vector(rec, spec, include_dynamic=include_dynamic)
            label = None
            if label_map:
                label = label_map.get(os.path.basename(f))
            if label is None and infer_from_name:
                label = label_from_filename(f)
            X.append(vec)
            y.append(label)
            used.append(os.path.basename(f))
            if verbose:
                print("  + %-32s -> %s" % (os.path.basename(f), label))
        except Exception as e:
            skipped.append((os.path.basename(f), str(e)))
            if verbose:
                print("  ! skip %-28s (%s)" % (os.path.basename(f), e))
    return names, X, y, used, skipped


def write_feature_csv(path, names, X, files, y=None):
    """Belgilar matritsasini CSV ga yozadi (statistik tahlil / sklearn uchun)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        header = ["file"] + (["label"] if y is not None else []) + names
        w.writerow(header)
        for i in range(len(files)):
            row = [files[i]] + ([y[i] if y[i] is not None else ""] if y is not None else [])
            row += ["%.6g" % v for v in X[i]]
            w.writerow(row)
    return path
