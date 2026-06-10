"""
timefreq.py — Vaqt-chastota (time-frequency) tahlili — ASOSIY ILMIY YANGILIK.

Klassik Welch/FFT spektri butun yozuvni BITTA o'rtacha spektrga aylantiradi va
signal STATSIONAR deb faraz qiladi. Lekin sportchining funksional holati
(diqqat, charchoq, qo'zg'alish) yuk yoki vazifa davomida VAQT bo'yicha
o'zgaradi. Bu modul aynan shu DINAMIKANI ochib beradi:

  1) STFT (Short-Time Fourier Transform) — vaqt bo'yicha siljiydigan oyna
     bilan spektrogramma (tez, ritm quvvatining vaqt qatorlari uchun).
  2) Morlet uzluksiz veyvlet almashtirishi (CWT) — chastota bo'yicha
     moslashuvchan ajratish (past chastotada vaqt aniqligi past, yuqori
     chastotada yuqori) — biosignallar uchun klassik tanlov.
  3) Ritmlar (delta..gamma) bo'yicha QUVVATNING VAQT QATORLARI va ulardan
     DINAMIK BELGILAR (o'rtacha, o'zgaruvchanlik, trend, maksimum) — bular
     ML uchun qo'shimcha, statik spektrda mavjud bo'lmagan ma'lumot beradi.

Hammasi SOF PYTHON (dsp.fft/ifft) ustida; numpy mavjud bo'lsa avtomatik
tezlashadi. Hisoblash og'ir bo'lmasligi uchun signal kerak bo'lganda
kamaytiriladi (decimation) — STFT standart, CWT vizualizatsiya uchun.
"""

import math
import cmath

from . import dsp
from . import config


# ---------------------------------------------------------------------------
# STFT — siljiydigan oyna spektrogrammasi
# ---------------------------------------------------------------------------
def stft(signal, fs, win_sec=1.0, overlap=0.5):
    """
    Qisqa vaqtli Furye almashtirishi.

    Qaytaradi: (times, freqs, power) bu yerda
      times  — har segment markazidagi vaqt (s),
      freqs  — chastota o'qi (Hz),
      power  — power[t_index][f_index] (bir tomonlama |X|^2).
    """
    n = len(signal)
    if n < 8 or fs <= 0:
        return [], [], []
    nperseg = max(8, int(win_sec * fs))
    nperseg = min(nperseg, n)
    nfft = dsp.next_pow2(nperseg)
    step = max(1, int(nperseg * (1.0 - overlap)))
    win = dsp.hann_window(nperseg)
    win_power = sum(w * w for w in win) or 1.0
    half = nfft // 2
    freqs = [k * fs / nfft for k in range(half + 1)]

    times, power = [], []
    start = 0
    while start + nperseg <= n:
        seg = signal[start:start + nperseg]
        seg = dsp.detrend_mean(seg)
        seg = [seg[i] * win[i] for i in range(nperseg)]
        p = dsp.rfft_power(seg, nfft)
        # bir tomonlama normalizatsiya (PSD ga mos)
        norm = fs * win_power
        row = []
        for k in range(half + 1):
            v = p[k] / norm
            if k != 0 and k != half:
                v *= 2.0
            row.append(v)
        power.append(row)
        times.append((start + nperseg / 2.0) / fs)
        start += step

    return times, freqs, power


# ---------------------------------------------------------------------------
# Morlet uzluksiz veyvlet almashtirishi (CWT) — FFT orqali konvolyutsiya
# ---------------------------------------------------------------------------
def morlet_cwt(signal, fs, freqs=None, w0=6.0):
    """
    Kompleks Morlet veyvleti bilan CWT (FFT-konvolyutsiya orqali).

    freqs — qiziqtirgan chastotalar (Hz). Berilmasa, 2..min(45,Nyquist) Hz
            oralig'ida ~40 ta logarifmik nuqta.
    w0    — Morlet markaziy parametri (odatda 6) — vaqt/chastota muvozanati.

    Qaytaradi: (freqs, power) bu yerda power[f_index][t] — har vaqt nuqtasidagi
    skalogramma quvvati (|W|^2). Uzunlik = signal uzunligi.
    """
    n = len(signal)
    if n < 8 or fs <= 0:
        return [], []
    nyq = fs / 2.0
    fmax = min(config.ANALYSIS_BAND[1], nyq * 0.95)
    fmin = max(config.ANALYSIS_BAND[0], 2.0)
    if freqs is None:
        m = 40
        lo, hi = math.log(fmin), math.log(fmax)
        freqs = [math.exp(lo + (hi - lo) * i / (m - 1)) for i in range(m)]

    # signalni FFT uchun 2 darajasiga to'ldirish
    nfft = dsp.next_pow2(n)
    x = dsp.detrend_mean(list(signal)) + [0.0] * (nfft - n)
    X = dsp.fft(x)

    # chastota o'qi (FFT bin -> Hz), to'liq (manfiy chastotalarsiz Morlet)
    fft_freqs = [(k * fs / nfft) if k <= nfft // 2 else ((k - nfft) * fs / nfft)
                 for k in range(nfft)]

    power = []
    for f0 in freqs:
        if f0 <= 0:
            power.append([0.0] * n)
            continue
        # chastota sohasidagi Morlet (analitik): faqat musbat chastotalarda
        sigma_f = f0 / w0
        psi = [0.0] * nfft
        norm = (math.pi ** -0.25)
        for k in range(nfft):
            fk = fft_freqs[k]
            if fk <= 0:
                continue
            # Morlet chastota sohasida: Gauss markazi f0
            arg = -0.5 * ((fk - f0) / sigma_f) ** 2
            psi[k] = norm * math.sqrt(2.0) * math.exp(arg)
        conv_spec = [X[k] * psi[k] for k in range(nfft)]
        w = dsp.ifft(conv_spec)
        power.append([abs(w[i]) ** 2 for i in range(n)])

    return freqs, power


# ---------------------------------------------------------------------------
# Ritmlar bo'yicha quvvatning VAQT QATORLARI (STFT asosida — tez)
# ---------------------------------------------------------------------------
def band_time_courses(signal, fs, win_sec=1.0, overlap=0.5, bands=None):
    """
    Har ritm (delta..gamma) uchun NISBIY quvvatning vaqt qatori.

    Qaytaradi: (times, {band: [rel_power_t, ...]})
    """
    bands = bands or config.BANDS
    times, freqs, power = stft(signal, fs, win_sec=win_sec, overlap=overlap)
    if not times:
        return [], {b: [] for b in bands}

    def integ(row, lo, hi):
        total = 0.0
        for k in range(len(freqs) - 1):
            f0, f1 = freqs[k], freqs[k + 1]
            if f1 < lo or f0 > hi:
                continue
            a, b = max(f0, lo), min(f1, hi)
            if b <= a:
                continue
            total += 0.5 * (row[k] + row[k + 1]) * (b - a)
        return total

    courses = {b: [] for b in bands}
    nyq = fs / 2.0
    for row in power:
        absb = {b: integ(row, lo, min(hi, nyq)) for b, (lo, hi) in bands.items()}
        tot = sum(absb.values()) or 1e-12
        for b in bands:
            courses[b].append(absb[b] / tot)
    return times, courses


# ---------------------------------------------------------------------------
# Dinamik belgilar — ML uchun (statik spektrda yo'q ma'lumot)
# ---------------------------------------------------------------------------
def _trend(y):
    """Vaqt qatorining chiziqli trend qiyaligi (eng kichik kvadratlar)."""
    n = len(y)
    if n < 2:
        return 0.0
    sx = sum(range(n)); sy = sum(y)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * y[i] for i in range(n))
    denom = (n * sxx - sx * sx) or 1.0
    return (n * sxy - sx * sy) / denom


def dynamic_features(rec, win_sec=1.0, overlap=0.5, max_channels=8):
    """
    Yozuv bo'yicha ritmlar quvvatining VAQT DINAMIKASIDAN belgilar ajratadi.

    Har ritm uchun (kanallar o'rtachasi):
      *_dyn_mean — vaqt bo'yicha o'rtacha nisbiy quvvat
      *_dyn_std  — o'zgaruvchanlik (beqarorlik / fluktuatsiya)
      *_dyn_trend— vaqt bo'yicha trend (oshish/kamayish qiyaligi)
      *_dyn_range— maksimal - minimal

    Bu belgilar statik spektrda mavjud emas — holatning DAVOMIDAGI
    o'zgarishini (masalan, charchoqning asta o'sishi) ifodalaydi.
    """
    chans = rec.channels[:max_channels] if rec.channels else []
    agg = {b: {"mean": [], "std": [], "trend": [], "range": []} for b in config.BANDS}

    for ch in chans:
        fs = rec.fs.get(ch, 0.0)
        sig = rec.signals.get(ch, [])
        if fs <= 0 or len(sig) < int(win_sec * fs) * 2:
            continue
        _t, courses = band_time_courses(sig, fs, win_sec=win_sec, overlap=overlap)
        for b in config.BANDS:
            y = courses.get(b, [])
            if len(y) < 2:
                continue
            agg[b]["mean"].append(dsp.mean(y))
            agg[b]["std"].append(dsp.std(y))
            agg[b]["trend"].append(_trend(y))
            agg[b]["range"].append(max(y) - min(y))

    feats = {}
    for b in config.BANDS:
        for stat in ("mean", "std", "trend", "range"):
            vals = agg[b][stat]
            feats["%s_dyn_%s" % (b, stat)] = (sum(vals) / len(vals)) if vals else 0.0
    return feats


def summarize_dynamics(rec, win_sec=1.0, overlap=0.5, max_channels=8):
    """Inson o'qishi uchun qisqa matnli dinamika xulosasi (hisobotga)."""
    feats = dynamic_features(rec, win_sec=win_sec, overlap=overlap,
                             max_channels=max_channels)
    lines = ["VAQT-CHASTOTA DINAMIKASI (ritmlar quvvatining vaqt bo'yicha o'zgarishi)",
             "-" * 60]
    for b in config.BANDS:
        m = feats["%s_dyn_mean" % b] * 100
        s = feats["%s_dyn_std" % b] * 100
        tr = feats["%s_dyn_trend" % b]
        arrow = "↑ oshmoqda" if tr > 1e-4 else ("↓ kamaymoqda" if tr < -1e-4 else "→ barqaror")
        lines.append("  %-6s o'rtacha=%5.1f%%  o'zgaruvchanlik=%5.1f%%  trend: %s"
                     % (config.BAND_LABELS[b], m, s, arrow))
    return "\n".join(lines), feats
