#!/usr/bin/env python3
"""
train_advanced.py — scikit-learn / SHAP / EEGNet bilan KUCHAYTIRILGAN o'qitish
(LOKAL kompyuterda, to'liq kutubxonalar bilan).

DIQQAT: bu skript scikit-learn (va ixtiyoriy shap, tensorflow) talab qiladi.
Sandboxda yoki kutubxonalarsiz muhitda SOF PYTHON variantdan foydalaning:
    python3 train_ai.py train <papka> [...]
Bu skript dissertatsiyaning yakuniy (lokal) bosqichi uchun — bir xil belgilar
ustida sklearn modellari, haqiqiy SHAP qiymatlari va EEGNet bilan ishlaydi.

Bosqichlar:
  1) eeg_engine bilan belgilar matritsasini tuzish (qurilma/format-agnostik).
  2) sklearn RandomForest / GradientBoosting / SVM ni cross-validatsiya bilan.
  3) SHAP bilan belgi hissalarini tushuntirish (mavjud bo'lsa).
  4) (ixtiyoriy) EEGNet ni xom epoxalarda o'qitish.

Misol:
    pip install scikit-learn shap tensorflow
    python3 train_advanced.py --data /path/EEG-signals --labels labels.csv --shap
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="sklearn/SHAP/EEGNet bilan o'qitish (lokal).")
    ap.add_argument("--data", required=True, help="EEG yozuvlar papkasi")
    ap.add_argument("--labels", help="labels.csv (file,label)")
    ap.add_argument("--infer-name", action="store_true")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--shap", action="store_true", help="SHAP tahlilini ishga tushirish")
    ap.add_argument("--eegnet", action="store_true", help="EEGNet (tensorflow) o'qitish")
    args = ap.parse_args()

    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        print("scikit-learn kerak: pip install scikit-learn numpy\nXato: %s" % e,
              file=sys.stderr)
        return 1

    from eeg_engine import dataset

    print("Belgilar matritsasi tuzilmoqda...")
    names, X, y, files, skipped = dataset.build_dataset(
        args.data, labels_csv=args.labels, infer_from_name=args.infer_name)
    X = [X[i] for i in range(len(y)) if y[i]]
    files2 = [files[i] for i in range(len(y)) if y[i]]
    y = [v for v in y if v]
    if len(X) < 5:
        print("Yorliqlangan yozuvlar juda kam.", file=sys.stderr)
        return 1
    X = np.asarray(X, dtype=float)

    models = {
        "RandomForest": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42)),
        "GradientBoosting": make_pipeline(
            StandardScaler(), GradientBoostingClassifier(random_state=42)),
        "SVM-RBF": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    }

    from collections import Counter
    min_class = min(Counter(y).values())
    k = max(2, min(args.cv, min_class))
    cvsplit = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    print("\n%d sinf, %d namuna, %d-fold CV\n" % (len(set(y)), len(y), k))
    best_name, best_score, best_model = None, -1, None
    for name, mdl in models.items():
        scores = cross_val_score(mdl, X, y, cv=cvsplit)
        print("  %-18s CV aniqlik: %.1f%% (+/- %.1f%%)"
              % (name, scores.mean() * 100, scores.std() * 100))
        if scores.mean() > best_score:
            best_name, best_score, best_model = name, scores.mean(), mdl

    print("\nEng yaxshi model: %s (%.1f%%). Butun ma'lumotda o'qitilmoqda..."
          % (best_name, best_score * 100))
    best_model.fit(X, y)

    if args.shap:
        try:
            import shap
            print("\nSHAP tahlili...")
            clf = best_model.steps[-1][1]
            Xs = best_model.steps[0][1].transform(X)
            explainer = shap.Explainer(clf, Xs)
            sv = explainer(Xs)
            import numpy as _np
            mean_abs = _np.abs(sv.values).mean(axis=tuple(range(sv.values.ndim - 1)))
            ranked = sorted(zip(names, mean_abs), key=lambda t: t[1], reverse=True)
            print("Eng muhim belgilar (SHAP):")
            for nm, val in ranked[:12]:
                print("  %-22s %.4f" % (nm, val))
        except Exception as e:
            print("SHAP o'tkazib yuborildi: %s" % e)

    if args.eegnet:
        try:
            from eeg_engine import eegnet
            pairs = list(zip([dataset.os.path.join(args.data, f) if not dataset.os.path.isabs(f) else f
                              for f in files2], y))
            print("\nEEGNet uchun epoxalar tayyorlanmoqda...")
            Xe, ye, ch = eegnet.build_epoch_dataset(pairs)
            print("Epoxalar:", Xe.shape, "| kanallar:", len(ch) if ch else 0)
            model = eegnet.build_eegnet(n_classes=len(set(ye)),
                                        chans=Xe.shape[1], samples=Xe.shape[2])
            eegnet.train(model, Xe, ye, epochs=80)
        except Exception as e:
            print("EEGNet o'tkazib yuborildi: %s" % e)

    print("\nTayyor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
