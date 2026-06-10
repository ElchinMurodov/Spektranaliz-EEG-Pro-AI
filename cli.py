#!/usr/bin/env python3
"""
cli.py — Spektranaliz EEG Pro dasturining buyruq qatori (CLI) interfeysi.

GUI dan tashqari, dasturni terminalda (va skriptlarda) ham ishlatish mumkin.

Foydalanish:
    python cli.py <fayl.edf|.bdf|.csv> [parametrlar]

Misollar:
    python cli.py data/0000007.EDF
    python cli.py data/synth_stress_100hz.edf --html hisobot.html
    python cli.py signal.csv --fs 256
    python cli.py data/synth_fokus_500hz.edf --target-fs 100
    python cli.py mashqdan_keyin.edf --baseline dam_olish.edf
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_engine import analyze_file


def main():
    p = argparse.ArgumentParser(
        description="EEG signallarini spektral tahlil qilib, funksional holatni aniqlaydi.")
    p.add_argument("file", help="EEG fayl yo'li (EDF/EDF+/BDF/BDF+/CSV)")
    p.add_argument("--fs", type=float, default=None, help="Namuna chastotasi (CSV uchun)")
    p.add_argument("--target-fs", type=float, default=None, help="Harmonizatsiya chastotasi (Hz)")
    p.add_argument("--no-notch", action="store_true", help="50/60 Hz tarmoq filtrini o'chirish")
    p.add_argument("--html", metavar="FAYL", help="HTML vizual hisobotni saqlash")
    p.add_argument("--pdf", metavar="FAYL", help="PDF grafikli hisobotni saqlash (Pillow kerak)")
    p.add_argument("--txt", metavar="FAYL", help="Matnli (TXT) hisobotni saqlash")
    p.add_argument("--baseline", metavar="FAYL", help="Tinch holat yozuvi (individual kalibrlash)")
    p.add_argument("--reader", default="auto", choices=["auto", "pyedflib", "mne", "pure"],
                   help="EDF/BDF o'qish usuli")
    p.add_argument("--timefreq", action="store_true",
                   help="Vaqt-chastota (wavelet/STFT) dinamikasi xulosasini ham chiqarish")
    p.add_argument("--model", metavar="FAYL",
                   help="O'qitilgan AI modeli (train_ai.py natijasi, model.json) bilan bashorat")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print("XATO: fayl topilmadi:", args.file)
        sys.exit(1)

    base = None
    if args.baseline:
        from eeg_engine import calibration
        base = calibration.compute_baseline(args.baseline)
        print("Kalibrlash: baseline '%s' dan hisoblandi.\n" % args.baseline)

    result = analyze_file(
        args.file, fs=args.fs, target_fs=args.target_fs,
        notch=not args.no_notch, html_path=args.html, pdf_path=args.pdf,
        txt_path=args.txt, baseline=base, prefer=args.reader)
    print(result["report"])
    if args.html:
        print("\nHTML hisobot saqlandi:", args.html)
    if args.pdf:
        print("PDF hisobot saqlandi:", args.pdf)
    if args.txt:
        print("TXT hisobot saqlandi:", args.txt)

    # --- Vaqt-chastota dinamikasi (Stage 3) ---
    if args.timefreq:
        from eeg_engine import loader, preprocessing, timefreq
        rec = loader.load(args.file, fs=args.fs, prefer=args.reader)
        preprocessing.preprocess(rec, target_fs=args.target_fs, notch=not args.no_notch)
        txt, _feats = timefreq.summarize_dynamics(rec)
        print("\n" + txt)

    # --- AI (mashinaviy o'qitish) bashorati (Stage 4) ---
    if args.model:
        from eeg_engine import loader, preprocessing, spectral, dataset, ml
        if not os.path.exists(args.model):
            print("\nXATO: AI modeli topilmadi:", args.model)
        else:
            bundle = ml.load_model(args.model)
            model = bundle["model"]; std = bundle["standardizer"]; names = bundle["feature_names"]
            rec = loader.load(args.file, fs=args.fs, prefer=args.reader)
            preprocessing.preprocess(rec, target_fs=args.target_fs, notch=not args.no_notch)
            spec = spectral.analyze_recording(rec)
            include_dynamic = names is not None and len(names) > len(dataset.STATIC_FEATURES)
            _n, vec = dataset.feature_vector(rec, spec, include_dynamic=include_dynamic)
            Xs = std.transform([vec]) if std else [vec]
            proba = model.predict_proba(Xs)[0]
            order = sorted(range(len(model.classes_)), key=lambda j: proba[j], reverse=True)
            print("\n" + "=" * 60)
            print("  AI (MASHINAVIY O'QITISH) BASHORATI")
            print("=" * 60)
            print("  AI HOLAT : %s (%.1f%%)"
                  % (model.classes_[order[0]], proba[order[0]] * 100))
            print("  " + "-" * 58)
            for j in order:
                print("    %-26s %5.1f%%" % (model.classes_[j], proba[j] * 100))
            print("  (Eslatma: bu AI modeli o'qitilgan yorliqlarga bog'liq; "
                  "qoidaviy natija bilan solishtiring.)")


if __name__ == "__main__":
    main()
