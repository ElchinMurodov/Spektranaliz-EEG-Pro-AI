"""
assessment.py — Umumiy funksional va ruhiy holat baholash (skrining).

Spektral belgilardan (features.extract_features natijasi) yuqori darajadagi,
inson o'qiy oladigan ko'rsatkichlarni 0-100 shkalada hisoblaydi:

  * Diqqat darajasi (attention)
  * Charchoq (mental fatigue)
  * Stress
  * Psixofiziologik zo'riqish (arousal / faollashuv)
  * Emotsional zo'riqish (frontal alfa asimmetriyasi asosida)
  * Miya plastisiteti (bilvosita proksi: iAPF + alfa + entropiya)
  * Umumiy funksional tayyorlik (kompozit)
  * Biofeedback / neyrofeedback protokoli tavsiyasi

DIQQAT: bu ko'rsatkichlar TIBBIY TASHXIS EMAS — faqat skrining indekslari.
Ilmiy asoslar: engagement (Pope va b., 1995), theta/beta (Lubar; Monastra),
mental fatigue (Borghini va b., 2014), FAA (Davidson; Coan & Allen, 2004),
iAPF (Klimesch, 1999), neyrofeedback (Sterman & Egner, 2006; Peniston, 1989).
"""

from . import config


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _scale(v, lo, hi):
    return 0.0 if hi == lo else _clamp((v - lo) / (hi - lo))


def _level(v):
    return "past" if v < 35 else ("o'rta" if v < 65 else "yuqori")


def assess(features):
    """Spektral belgilar (dict) -> funksional/ruhiy holat indekslari (dict)."""
    f = features
    d = f.get("rp_delta", 0.0); th = f.get("rp_theta", 0.0)
    a = f.get("rp_alpha", 0.0); b = f.get("rp_beta", 0.0); g = f.get("rp_gamma", 0.0)
    tbr = f.get("ratio_theta_beta", th / (b + 1e-9))
    bar = f.get("ratio_beta_alpha", b / (a + 1e-9))
    eng = f.get("engagement", b / (a + th + 1e-9))
    iapf = f.get("iapf", 10.0)
    entropy = f.get("spectral_entropy", 0.7)
    faa = f.get("faa")

    attention = 100 * (0.55 * _scale(eng, 0.3, 1.0) + 0.45 * (1 - _scale(tbr, 1.0, 4.0)))
    fatigue = 100 * (0.45 * _scale(tbr, 1.2, 5.0) + 0.35 * _scale(d + th, 0.45, 0.80)
                     + 0.20 * (1 - _scale(b, 0.10, 0.35)))
    stress = 100 * (0.45 * _scale(bar, 0.9, 2.8) + 0.30 * _scale(g, 0.05, 0.20)
                    + 0.25 * (1 - _scale(a, 0.15, 0.40)))
    arousal = 100 * _scale((b + g) / (d + th + 1e-9), 0.15, 0.90)
    if faa is not None:
        emotional = 100 * (0.65 * (1 - _scale(faa, -0.6, 0.6)) + 0.35 * _scale(bar, 0.9, 2.8))
    else:
        emotional = 100 * _scale(bar, 0.9, 2.8)
    plasticity = 100 * (0.40 * _scale(iapf, 8.0, 11.5) + 0.35 * _scale(a, 0.10, 0.40)
                        + 0.25 * _scale(entropy, 0.50, 0.90))
    readiness = (0.35 * attention + 0.25 * (100 - fatigue)
                 + 0.25 * (100 - stress) + 0.15 * (100 - arousal))

    return {
        "attention": round(_clamp(attention, 0, 100), 1),
        "fatigue": round(_clamp(fatigue, 0, 100), 1),
        "stress": round(_clamp(stress, 0, 100), 1),
        "psychophysiological_tension": round(_clamp(arousal, 0, 100), 1),
        "emotional_tension": round(_clamp(emotional, 0, 100), 1),
        "brain_plasticity": round(_clamp(plasticity, 0, 100), 1),
        "overall_readiness": round(_clamp(readiness, 0, 100), 1),
        "faa_available": faa is not None,
    }


def neurofeedback_protocol(s, features):
    """Holatga qarab biofeedback/neyrofeedback protokoli tavsiyalari (ro'yxat)."""
    r = []
    if s["attention"] < 45 or s["fatigue"] > 60:
        r.append("Diqqat/charchoq: SMR (12-15 Hz) va past Beta ^ , Theta v "
                 "(Sterman & Egner; Lubar)")
    if s["stress"] > 60 or s["psychophysiological_tension"] > 65:
        r.append("Stress/zo'riqish: Alpha (8-12 Hz) ^ / Alpha-Theta, Beta v "
                 "(Peniston & Kulkosky)")
    if (features.get("rp_delta", 0) + features.get("rp_theta", 0)) > 0.6 \
            and s["psychophysiological_tension"] < 35:
        r.append("Uyquchanlik: Beta/SMR ^ , Delta v (faollashtirish)")
    if not r:
        r.append("Holat muvozanatli - joriy band qiymatlarini saqlash (maintenance)")
    return r


def _bar(v, width=18):
    n = int(round(_clamp(v / 100.0) * width))
    return "#" * n + "-" * (width - n)


def assessment_section(features, fancy=True):
    """build_report uchun matn satrlari ro'yxatini qaytaradi."""
    s = assess(features)
    sub = "-" * 60
    L = []
    A = L.append
    A("")
    A("UMUMIY FUNKSIONAL VA RUHIY HOLAT (0-100, skrining)")
    A(sub)
    items = [
        ("Diqqat darajasi", s["attention"]),
        ("Charchoq", s["fatigue"]),
        ("Stress", s["stress"]),
        ("Psixofiziologik zo'riqish", s["psychophysiological_tension"]),
        ("Emotsional zo'riqish", s["emotional_tension"]),
        ("Miya plastisiteti (proksi)", s["brain_plasticity"]),
    ]
    for name, val in items:
        A("  %-28s %5.1f  [%-5s] %s" % (name, val, _level(val), _bar(val) if fancy else ""))
    A("  %-28s %5.1f / 100" % (">> UMUMIY FUNKSIONAL TAYYORLIK", s["overall_readiness"]))
    if not s["faa_available"]:
        A("  (eslatma: FAA uchun F3/F4 yetarli emas - emotsional indeks beta asosida)")
    A("")
    A("BIOFEEDBACK / NEYROFEEDBACK TAVSIYASI")
    A(sub)
    for rec in neurofeedback_protocol(s, features):
        A("  - " + rec)
    A("  Joriy band (feedback bazasi): Delta=%.0f%% Theta=%.0f%% Alpha=%.0f%% "
      "Beta=%.0f%% Gamma=%.0f%%" % (
          features.get("rp_delta", 0) * 100, features.get("rp_theta", 0) * 100,
          features.get("rp_alpha", 0) * 100, features.get("rp_beta", 0) * 100,
          features.get("rp_gamma", 0) * 100))
    return L
