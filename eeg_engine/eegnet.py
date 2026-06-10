"""
eegnet.py — EEGNet (chuqur o'qitish) — XOM EEG epoxalari uchun (LOKAL ishlatish).

DIQQAT: bu modul tensorflow/keras kutubxonalarini talab qiladi va katta
hisoblash resurslari (yaxshisi GPU) bilan LOKAL kompyuterda ishlatish uchun.
Sof Python Random Forest (ml.py + train_ai.py) esa har joyda ishlaydi va kam
ma'lumotda (60+ yozuv) ASOSIY tavsiya etilgan modeldir. EEGNet faqat yozuvlar
EPOXALARGA bo'linib, namuna soni minglarga yetganda kuchli bo'ladi.

EEGNet — kam parametrli, EEG uchun maxsus ixchamlashtirilgan CNN
(Lawhern va b., 2018). Bu yerda klassik arxitektura keltirilgan.

Foydalanish (lokal):
    from eeg_engine import eegnet, loader, preprocessing
    X, y, ch = eegnet.build_epoch_dataset(file_label_pairs, epoch_sec=2.0)
    model = eegnet.build_eegnet(n_classes=len(set(y)),
                                chans=X.shape[1], samples=X.shape[2])
    eegnet.train(model, X, y)
"""

# Bu importlar faqat funksiya chaqirilganda tekshiriladi (modulni import
# qilishning o'zi tensorflow talab qilmaydi).
def _require_tf():
    try:
        import tensorflow as tf  # noqa
        return tf
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "EEGNet uchun tensorflow kerak. Lokal kompyuterda o'rnating:\n"
            "    pip install tensorflow scikit-learn\n"
            "Sandbox/oddiy holatda Random Forest (train_ai.py) dan foydalaning."
        ) from e


def build_epoch_dataset(file_label_pairs, epoch_sec=2.0, target_fs=128.0,
                        channels=None):
    """
    Yozuvlarni teng uzunlikdagi EPOXALARGA bo'lib, EEGNet uchun X, y tayyorlaydi.

    file_label_pairs : [(path, label), ...]
    Qaytaradi: (X, y, channels)
      X — shakli (n_epochs, n_channels, n_samples) numpy massiv
      y — yorliqlar ro'yxati (epox darajasida)
    Har yozuv bir nechta epoxaga aylanadi -> namuna soni keskin ortadi
    (deep learning uchun zarur).
    """
    import numpy as np
    from . import loader, preprocessing

    epochs, labels = [], []
    common_ch = channels
    for path, label in file_label_pairs:
        rec = loader.load(path)
        preprocessing.preprocess(rec, target_fs=target_fs)
        # umumiy kanallar to'plamini aniqlash (qurilmalar har xil bo'lishi mumkin)
        if common_ch is None:
            common_ch = list(rec.channels)
        chs = [c for c in common_ch if c in rec.signals]
        if not chs:
            continue
        fs = rec.fs.get(chs[0], target_fs)
        n_samp = int(epoch_sec * fs)
        if n_samp < 8:
            continue
        # kanallararo eng qisqa signal uzunligi
        length = min(len(rec.signals[c]) for c in chs)
        n_epochs = length // n_samp
        for e in range(n_epochs):
            seg = [rec.signals[c][e * n_samp:(e + 1) * n_samp] for c in chs]
            if all(len(s) == n_samp for s in seg):
                epochs.append(seg)
                labels.append(label)
    X = np.array(epochs, dtype="float32")  # (n, chans, samples)
    return X, labels, common_ch


def build_eegnet(n_classes, chans, samples, dropout=0.5, kern_length=64,
                 F1=8, D=2, F2=16):
    """
    EEGNet arxitekturasi (Keras). Kirish: (chans, samples, 1).
    """
    tf = _require_tf()
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, BatchNormalization, DepthwiseConv2D, Activation,
        AveragePooling2D, SeparableConv2D, Dropout, Flatten, Dense)
    from tensorflow.keras.constraints import max_norm

    inp = Input(shape=(chans, samples, 1))
    # 1) Vaqtli (temporal) konvolyutsiya
    x = Conv2D(F1, (1, kern_length), padding="same", use_bias=False)(inp)
    x = BatchNormalization()(x)
    # 2) Fazoviy (depthwise) konvolyutsiya — kanallar bo'ylab
    x = DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=D,
                        depthwise_constraint=max_norm(1.0))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout)(x)
    # 3) Ajratiladigan (separable) konvolyutsiya
    x = SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout)(x)
    # 4) Tasnif
    x = Flatten()(x)
    x = Dense(n_classes, kernel_constraint=max_norm(0.25))(x)
    out = Activation("softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def train(model, X, y, epochs=100, batch_size=16, val_split=0.2):
    """EEGNet ni o'qitadi. y — matn yorliqlar (avtomatik raqamga aylantiriladi)."""
    tf = _require_tf()
    import numpy as np
    classes = sorted(set(y))
    cidx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cidx[v] for v in y])
    Xr = X[..., np.newaxis]  # (n, chans, samples, 1)
    hist = model.fit(Xr, yi, epochs=epochs, batch_size=batch_size,
                     validation_split=val_split, verbose=2)
    return hist, classes
