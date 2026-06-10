#!/usr/bin/env python3
"""
train_ai.py — Sportchi EEG holatini NAZORATLI O'QITISH (AI) bilan aniqlash CLI.

Bu vosita qoidaviy klassifikatordan (eeg_engine/classifier.py) farqli o'laroq,
yorliqlangan yozuvlardan funksional holatni O'RGANADI. Belgilar statik spektral
+ dinamik vaqt-chastota belgilaridan iborat (dataset.py). Model — sof Python
Random Forest (eeg_engine/ml.py), numpy/sklearn shart emas.

Qurilma/format moslik: loader.py CONTEC KT-88 (16 ch), Нейротех "Компакт-нейро"
(21 ch) va ixtiyoriy kanalli .edf/.bdf/.csv yozuvlarni o'qiydi; belgilar
kanallar o'rtachasi sifatida olinadi, shuning uchun turli qurilmalar bitta
modelda birga ishlatiladi.

Buyruqlar:
  features  <papka> [--labels labels.csv | --infer-name] --out feats.csv
  train     <papka> [--labels labels.csv | --infer-name] [--out model.json] [--cv 5]
  cluster   <papka> [-k 3]
  predict   <fayl>  --model model.json

Misollar:
  # Sun'iy (yorliqli) ma'lumotda modelni o'qitish va tekshirish:
  python3 train_ai.py train data --infer-name --out model.json --cv 5

  # Haqiqiy sportchilar (labels.csv: file,label) bo'yicha:
  python3 train_ai.py train /path/EEG-signals --labels labels.csv --out model.json

  # Yorliq bo'lmasa — klasterlash (guruhlarni topish):
  python3 train_ai.py cluster /path/EEG-signals -k 4

  # Yangi yozuvni bashorat qilish:
  python3 train_ai.py predict newrec.edf --model model.json
"""

import os
import sys
import argparse

from eeg_engine import dataset, ml, loader, preprocessing, spectral


def _filter_labeled(X, y, files):
    fx, fy, ff = [], [], []
    for i in range(len(y)):
        if y[i] is not None and y[i] != "":
            fx.append(X[i]); fy.append(y[i]); ff.append(files[i])
    return fx, fy, ff


def cmd_features(args):
    names, X, y, files, skipped = dataset.build_dataset(
        args.path, labels_csv=args.labels, infer_from_name=args.infer_name,
        include_dynamic=not args.no_dynamic)
    has_labels = any(v is not None for v in y)
    dataset.write_feature_csv(args.out, names, X, files, y if has_labels else None)
    print("\nBelgilar matritsasi yozildi -> %s" % args.out)
    print("Yozuvlar: %d | Belgilar: %d | O'tkazib yuborilgan: %d"
          % (len(files), len(names), len(skipped)))
    return 0


def cmd_train(args):
    print("Ma'lumot tayyorlanmoqda...")
    names, X, y, files, skipped = dataset.build_dataset(
        args.path, labels_csv=args.labels, infer_from_name=args.infer_name,
        include_dynamic=not args.no_dynamic)
    X, y, files = _filter_labeled(X, y, files)
    if len(X) < 4:
        print("XATO: yorliqlangan yozuvlar juda kam (%d). labels.csv yoki "
              "--infer-name ni tekshiring." % len(X), file=sys.stderr)
        return 1

    classes = sorted(set(y))
    print("\nYorliqlangan yozuvlar: %d | Sinflar (%d): %s"
          % (len(X), len(classes), ", ".join(classes)))
    counts = {c: y.count(c) for c in classes}
    print("Sinf taqsimoti:", counts)

    # standartlashtirish
    std = ml.Standardizer().fit(X)
    Xs = std.transform(X)

    # cross-validatsiya
    k = min(args.cv, min(counts.values())) if counts else args.cv
    if k < 2:
        k = 2
    print("\n%d-fold stratifikatsiyalangan cross-validatsiya..." % k)

    def factory():
        return ml.RandomForest(n_trees=args.trees, max_depth=args.depth, seed=args.seed)

    cv = ml.cross_validate(factory, Xs, y, k=k, seed=args.seed)
    print("  Fold aniqliklari:", ["%.2f" % a for a in cv["fold_accuracies"]])
    print("  O'RTACHA ANIQLIK : %.1f%%" % (cv["mean_accuracy"] * 100))

    labels, cm = ml.confusion_matrix(cv["y_true"], cv["y_pred"], labels=classes)
    print("\nChalkashlik matritsasi (qator=haqiqiy, ustun=bashorat):")
    w = max(len(l) for l in labels)
    print(" " * (w + 2) + " ".join("%6.6s" % l for l in labels))
    for i, l in enumerate(labels):
        print("  %-*s " % (w, l) + " ".join("%6d" % v for v in cm[i]))

    # yakuniy model (barcha ma'lumotda)
    print("\nYakuniy modelni butun ma'lumotda o'qitish...")
    model = ml.RandomForest(n_trees=args.trees, max_depth=args.depth, seed=args.seed)
    model.fit(Xs, y)

    # belgi muhimligi (permutation — SHAP-ga o'xshash)
    base, imp = ml.permutation_importance(model, Xs, y, feature_names=names,
                                          repeats=args.imp_repeats, seed=args.seed)
    print("\nBELGI MUHIMLIGI (permutation importance, eng muhim 12 ta):")
    for nm, val in imp[:12]:
        bar = "#" * int(max(0.0, val) * 100)
        print("  %-22s %+.4f %s" % (nm, val, bar))

    if args.out:
        ml.save_model(model, args.out, feature_names=names, standardizer=std,
                      meta={"classes": classes, "cv_accuracy": cv["mean_accuracy"],
                            "n_samples": len(X)})
        print("\nModel saqlandi -> %s" % args.out)
    return 0


def cmd_cluster(args):
    print("Ma'lumot tayyorlanmoqda (yorliqsiz)...")
    names, X, y, files, skipped = dataset.build_dataset(
        args.path, include_dynamic=not args.no_dynamic)
    if len(X) < args.k:
        print("XATO: yozuvlar klaster sonidan kam.", file=sys.stderr)
        return 1
    std = ml.Standardizer().fit(X)
    Xs = std.transform(X)
    km = ml.KMeans(k=args.k, seed=args.seed).fit(Xs)
    print("\nKMeans (k=%d) | inersiya=%.2f" % (args.k, km.inertia_))

    from eeg_engine import classifier
    groups = {}
    for i, lab in enumerate(km.labels_):
        groups.setdefault(lab, []).append(i)

    band_idx = {b: dataset.STATIC_FEATURES.index("rp_" + b)
                for b in ("delta", "theta", "alpha", "beta", "gamma")}

    for c in sorted(groups):
        members = groups[c]
        # klaster bo'yicha o'rtacha (xom, standartlashtirilmagan) belgilar
        meanvec = [sum(X[i][j] for i in members) / len(members)
                   for j in range(len(names))]
        # statik qism -> features dict -> qoidaviy taxminiy holat
        fdict = {dataset.STATIC_FEATURES[j]: meanvec[j]
                 for j in range(len(dataset.STATIC_FEATURES))}
        try:
            suggested = classifier.classify(fdict)["state"]
        except Exception:
            suggested = "?"
        print("\n" + "=" * 60)
        print("  KLASTER %d  |  %d ta yozuv  |  taxminiy holat: %s"
              % (c, len(members), suggested))
        print("  " + "-" * 56)
        prof = "  ".join("%s=%.0f%%" % (b.capitalize(), meanvec[band_idx[b]] * 100)
                         for b in ("delta", "theta", "alpha", "beta", "gamma"))
        print("  Spektral profil: " + prof)
        for i in members:
            print("     - %s" % files[i])

    if args.pca:
        coords = ml.pca_2d(Xs)
        import csv as _csv
        with open(args.pca, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["file", "cluster", "pc1", "pc2"])
            for i in range(len(files)):
                w.writerow([files[i], km.labels_[i],
                            "%.4f" % coords[i][0], "%.4f" % coords[i][1]])
        print("\n2D PCA koordinatalari yozildi -> %s "
              "(lokalda scatter-plot chizish uchun)" % args.pca)

    print("\nKEYINGI QADAM: yuqoridagi 'taxminiy holat' va spektral profilga "
          "qarab, har klasterni nevrolog sifatida nomlang (masalan labels.csv "
          "da). So'ng 'train' bilan nazoratli model quramiz.")
    return 0


def cmd_predict(args):
    bundle = ml.load_model(args.model)
    model = bundle["model"]; std = bundle["standardizer"]
    names = bundle["feature_names"]
    rec = loader.load(args.path)
    preprocessing.preprocess(rec)
    spec = spectral.analyze_recording(rec)
    include_dynamic = names is not None and len(names) > len(dataset.STATIC_FEATURES)
    _n, vec = dataset.feature_vector(rec, spec, include_dynamic=include_dynamic)
    Xs = std.transform([vec]) if std else [vec]
    proba = model.predict_proba(Xs)[0]
    order = sorted(range(len(model.classes_)), key=lambda j: proba[j], reverse=True)
    print("Fayl: %s" % os.path.basename(args.path))
    print("BASHORAT QILINGAN HOLAT: %s (%.1f%%)"
          % (model.classes_[order[0]], proba[order[0]] * 100))
    print("\nBarcha sinflar bo'yicha ehtimollik:")
    for j in order:
        print("  %-26s %5.1f%%" % (model.classes_[j], proba[j] * 100))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="train_ai",
                                description="Sportchi EEG holati uchun AI (Random Forest).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("features", help="belgilar matritsasini CSV ga eksport")
    pf.add_argument("path")
    pf.add_argument("--labels"); pf.add_argument("--infer-name", action="store_true")
    pf.add_argument("--no-dynamic", action="store_true")
    pf.add_argument("--out", default="features.csv")
    pf.set_defaults(func=cmd_features)

    pt = sub.add_parser("train", help="Random Forest modelini o'qitish + CV")
    pt.add_argument("path")
    pt.add_argument("--labels"); pt.add_argument("--infer-name", action="store_true")
    pt.add_argument("--no-dynamic", action="store_true")
    pt.add_argument("--out"); pt.add_argument("--cv", type=int, default=5)
    pt.add_argument("--trees", type=int, default=60)
    pt.add_argument("--depth", type=int, default=8)
    pt.add_argument("--imp-repeats", type=int, default=5)
    pt.add_argument("--seed", type=int, default=42)
    pt.set_defaults(func=cmd_train)

    pc = sub.add_parser("cluster", help="nazoratsiz klasterlash (KMeans)")
    pc.add_argument("path")
    pc.add_argument("-k", type=int, default=3)
    pc.add_argument("--no-dynamic", action="store_true")
    pc.add_argument("--pca", metavar="FAYL", help="2D PCA koordinatalarini CSV ga yozish")
    pc.add_argument("--seed", type=int, default=42)
    pc.set_defaults(func=cmd_cluster)

    pp = sub.add_parser("predict", help="yangi yozuvni bashorat qilish")
    pp.add_argument("path"); pp.add_argument("--model", required=True)
    pp.set_defaults(func=cmd_predict)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
