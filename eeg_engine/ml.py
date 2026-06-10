"""
ml.py — Mashinaviy o'qitish (ML) yadrosi — SOF PYTHON (tashqi kutubxonasiz).

Bu modul qoidaviy klassifikatorni (classifier.py) NAZORATLI O'QITISH bilan
almashtirish/to'ldirish imkonini beradi. Yorliqlangan (ground-truth) EEG
yozuvlari bo'lsa (masalan: tinch / yukdan keyin / vazifa paytida), model
spektral + vaqt-chastota belgilaridan funksional holatni O'RGANADI.

Tarkibi (hammasi sof Python, numpy shart emas):
  * Standardizer        — belgilarni z-normallashtirish (mean/std)
  * KMeans              — nazoratsiz klasterlash (yorliq bo'lmasa)
  * DecisionTree (CART) — Gini bo'linishi bilan qaror daraxti
  * RandomForest        — bootstrap + tasodifiy belgilar (ansambl)
  * stratified_kfold    — sinflarni saqlovchi cross-validatsiya
  * permutation_importance — belgi muhimligi (SHAP-ga o'xshash, model-agnostik)
  * JSON save/load      — modelni faylga saqlash/yuklash

numpy mavjud bo'lsa ham, bu amalga oshirish unga bog'liq emas — har joyda
ishlaydi. Katta datasetlar uchun sklearn ekvivalenti train_ai.py da ko'rsatilgan.
"""

import math
import json
import random


# ===========================================================================
# Standartlashtirish (z-score)
# ===========================================================================
class Standardizer:
    def __init__(self):
        self.mean = []
        self.std = []

    def fit(self, X):
        n = len(X)
        d = len(X[0]) if n else 0
        self.mean = [0.0] * d
        self.std = [1.0] * d
        for j in range(d):
            col = [X[i][j] for i in range(n)]
            m = sum(col) / n
            var = sum((v - m) ** 2 for v in col) / n if n else 0.0
            self.mean[j] = m
            self.std[j] = math.sqrt(var) or 1.0
        return self

    def transform(self, X):
        return [[(row[j] - self.mean[j]) / self.std[j] for j in range(len(row))]
                for row in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ===========================================================================
# KMeans (nazoratsiz klasterlash)
# ===========================================================================
class KMeans:
    def __init__(self, k=3, iters=100, seed=42):
        self.k = k
        self.iters = iters
        self.seed = seed
        self.centroids = []

    def fit(self, X):
        rng = random.Random(self.seed)
        n = len(X)
        d = len(X[0])
        # k-means++ boshlang'ich
        first = rng.randrange(n)
        centroids = [list(X[first])]
        for _ in range(1, self.k):
            d2 = []
            for x in X:
                d2.append(min(_dist2(x, c) for c in centroids))
            tot = sum(d2) or 1.0
            r = rng.random() * tot
            acc = 0.0
            for i, val in enumerate(d2):
                acc += val
                if acc >= r:
                    centroids.append(list(X[i]))
                    break
            else:
                centroids.append(list(X[rng.randrange(n)]))

        labels = [0] * n
        for _ in range(self.iters):
            changed = False
            for i, x in enumerate(X):
                best = min(range(self.k), key=lambda c: _dist2(x, centroids[c]))
                if best != labels[i]:
                    labels[i] = best
                    changed = True
            for c in range(self.k):
                members = [X[i] for i in range(n) if labels[i] == c]
                if members:
                    centroids[c] = [sum(m[j] for m in members) / len(members)
                                    for j in range(d)]
            if not changed:
                break
        self.centroids = centroids
        self.labels_ = labels
        self.inertia_ = sum(_dist2(X[i], centroids[labels[i]]) for i in range(n))
        return self

    def predict(self, X):
        return [min(range(self.k), key=lambda c: _dist2(x, self.centroids[c]))
                for x in X]


def _dist2(a, b):
    return sum((a[j] - b[j]) ** 2 for j in range(len(a)))


# ===========================================================================
# PCA (2D proyeksiya) — klasterlarni vizualizatsiya qilish uchun (sof Python)
# ===========================================================================
def _jacobi_eigen(A, iters=200, tol=1e-12):
    """Simmetrik matritsa uchun Jacobi eigen-dekompozitsiyasi.
    Qaytaradi: (eigenvalues, eigenvectors) — ustunlar xususiy vektorlar."""
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > off:
                    off = abs(a[i][j]); p, q = i, j
        if off < tol:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        phi = 0.5 * math.atan2(2.0 * apq, app - aqq)
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(n):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def pca_2d(X):
    """Belgilar matritsasini 2 ta asosiy komponentga proyeksiyalaydi.
    Qaytaradi: [(pc1, pc2), ...] — har yozuv uchun 2D koordinata."""
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])
    mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    Xc = [[X[i][j] - mean[j] for j in range(d)] for i in range(n)]
    denom = (n - 1) if n > 1 else 1
    cov = [[0.0] * d for _ in range(d)]
    for aa in range(d):
        for bb in range(aa, d):
            sv = sum(Xc[i][aa] * Xc[i][bb] for i in range(n)) / denom
            cov[aa][bb] = sv; cov[bb][aa] = sv
    evals, evecs = _jacobi_eigen(cov)
    order = sorted(range(d), key=lambda j: evals[j], reverse=True)[:2]
    pc1 = [evecs[r][order[0]] for r in range(d)]
    pc2 = [evecs[r][order[1]] for r in range(d)]
    coords = []
    for i in range(n):
        x1 = sum(Xc[i][r] * pc1[r] for r in range(d))
        x2 = sum(Xc[i][r] * pc2[r] for r in range(d))
        coords.append((x1, x2))
    return coords


# ===========================================================================
# CART qaror daraxti (Gini)
# ===========================================================================
class _Node:
    __slots__ = ("feature", "thresh", "left", "right", "pred", "proba")

    def __init__(self):
        self.feature = None
        self.thresh = None
        self.left = None
        self.right = None
        self.pred = None
        self.proba = None


class DecisionTree:
    def __init__(self, max_depth=8, min_samples=2, n_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features      # tasodifiy belgilar soni (RF uchun)
        self.seed = seed
        self.root = None
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        self._cidx = {c: i for i, c in enumerate(self.classes_)}
        self._rng = random.Random(self.seed)
        self.root = self._build(X, y, 0)
        return self

    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        counts = {}
        for v in y:
            counts[v] = counts.get(v, 0) + 1
        imp = 1.0
        for c in counts.values():
            p = c / n
            imp -= p * p
        return imp

    def _leaf(self, y):
        node = _Node()
        counts = {c: 0 for c in self.classes_}
        for v in y:
            counts[v] += 1
        n = len(y) or 1
        node.pred = max(counts, key=counts.get)
        node.proba = [counts[c] / n for c in self.classes_]
        return node

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples
                or len(set(y)) == 1):
            return self._leaf(y)

        d = len(X[0])
        feats = list(range(d))
        if self.n_features and self.n_features < d:
            feats = self._rng.sample(feats, self.n_features)

        best_gain, best_f, best_t = 0.0, None, None
        parent_imp = self._gini(y)
        n = len(y)
        classes = self.classes_
        for f in feats:
            # belgini bir marta saralab, chegaralarni bitta o'tishda inkremental
            # Gini bilan baholash (O(n) har belgi uchun) — tezlashtirilgan CART
            order = sorted(range(n), key=lambda i: X[i][f])
            sv = [X[i][f] for i in order]
            sy = [y[i] for i in order]
            left = {c: 0 for c in classes}
            right = {c: 0 for c in classes}
            for c in sy:
                right[c] += 1
            nl = 0
            for j in range(n - 1):
                c = sy[j]
                left[c] += 1
                right[c] -= 1
                nl += 1
                if sv[j] == sv[j + 1]:
                    continue
                nr = n - nl
                gl = 1.0 - sum((left[c] / nl) ** 2 for c in classes)
                gr = 1.0 - sum((right[c] / nr) ** 2 for c in classes)
                imp = (nl * gl + nr * gr) / n
                gain = parent_imp - imp
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = 0.5 * (sv[j] + sv[j + 1])

        if best_f is None:
            return self._leaf(y)

        lX, lY, rX, rY = [], [], [], []
        for k in range(n):
            if X[k][best_f] <= best_t:
                lX.append(X[k]); lY.append(y[k])
            else:
                rX.append(X[k]); rY.append(y[k])

        node = _Node()
        node.feature = best_f
        node.thresh = best_t
        node.left = self._build(lX, lY, depth + 1)
        node.right = self._build(rX, rY, depth + 1)
        return node

    def _predict_node(self, node, x):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.thresh else node.right
        return node

    def predict(self, X):
        return [self._predict_node(self.root, x).pred for x in X]

    def predict_proba(self, X):
        return [self._predict_node(self.root, x).proba for x in X]


# ===========================================================================
# Random Forest (ansambl)
# ===========================================================================
class RandomForest:
    def __init__(self, n_trees=60, max_depth=8, min_samples=2,
                 n_features="sqrt", seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.seed = seed
        self.trees = []
        self.classes_ = []

    def _nfeat(self, d):
        if self.n_features == "sqrt":
            return max(1, int(math.sqrt(d)))
        if self.n_features == "log2":
            return max(1, int(math.log(d, 2)))
        if isinstance(self.n_features, int):
            return min(self.n_features, d)
        return d

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        n = len(X)
        d = len(X[0])
        nf = self._nfeat(d)
        rng = random.Random(self.seed)
        self.trees = []
        for t in range(self.n_trees):
            idx = [rng.randrange(n) for _ in range(n)]   # bootstrap
            bX = [X[i] for i in idx]
            bY = [y[i] for i in idx]
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples=self.min_samples,
                                n_features=nf, seed=rng.randrange(1 << 30))
            tree.fit(bX, bY)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        agg = [[0.0] * len(self.classes_) for _ in range(len(X))]
        cidx = {c: i for i, c in enumerate(self.classes_)}
        for tree in self.trees:
            probs = tree.predict_proba(X)
            tmap = [cidx[c] for c in tree.classes_]
            for i in range(len(X)):
                for j, p in enumerate(probs[i]):
                    agg[i][tmap[j]] += p
        for i in range(len(X)):
            s = sum(agg[i]) or 1.0
            agg[i] = [v / s for v in agg[i]]
        return agg

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[max(range(len(p)), key=lambda j: p[j])] for p in proba]


# ===========================================================================
# Cross-validatsiya va metrikalar
# ===========================================================================
def stratified_kfold(y, k=5, seed=42):
    """Sinflarni taqsimlovchi k-fold indekslarini qaytaradi: [(train, test), ...]."""
    rng = random.Random(seed)
    by_class = {}
    for i, c in enumerate(y):
        by_class.setdefault(c, []).append(i)
    for c in by_class:
        rng.shuffle(by_class[c])
    folds = [[] for _ in range(k)]
    for c, idxs in by_class.items():
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    splits = []
    for f in range(k):
        test = folds[f]
        train = [i for ff in range(k) if ff != f for i in folds[ff]]
        if test and train:
            splits.append((train, test))
    return splits


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    labels = labels or sorted(set(y_true) | set(y_pred))
    idx = {c: i for i, c in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for a, b in zip(y_true, y_pred):
        m[idx[a]][idx[b]] += 1
    return labels, m


def cross_validate(model_factory, X, y, k=5, seed=42):
    """model_factory() -> yangi model. Har fold uchun aniqlikni qaytaradi."""
    splits = stratified_kfold(y, k=k, seed=seed)
    accs, all_true, all_pred = [], [], []
    for train, test in splits:
        m = model_factory()
        m.fit([X[i] for i in train], [y[i] for i in train])
        pred = m.predict([X[i] for i in test])
        truth = [y[i] for i in test]
        accs.append(accuracy(truth, pred))
        all_true.extend(truth); all_pred.extend(pred)
    return {
        "fold_accuracies": accs,
        "mean_accuracy": sum(accs) / len(accs) if accs else 0.0,
        "y_true": all_true,
        "y_pred": all_pred,
    }


# ===========================================================================
# Belgi muhimligi (permutation importance — SHAP-ga o'xshash, model-agnostik)
# ===========================================================================
def permutation_importance(model, X, y, feature_names=None, repeats=5, seed=42):
    """
    Har belgini tasodifiy aralashtirib, aniqlik qanchalik tushishini o'lchaydi.
    Katta tushish -> belgi muhimroq. (SHAP bilan bir xil g'oya: hissani o'lchash.)
    """
    rng = random.Random(seed)
    base = accuracy(y, model.predict(X))
    d = len(X[0])
    names = feature_names or ["f%d" % j for j in range(d)]
    importances = []
    for j in range(d):
        drops = []
        for _ in range(repeats):
            Xp = [row[:] for row in X]
            col = [Xp[i][j] for i in range(len(Xp))]
            rng.shuffle(col)
            for i in range(len(Xp)):
                Xp[i][j] = col[i]
            drops.append(base - accuracy(y, model.predict(Xp)))
        importances.append((names[j], sum(drops) / len(drops)))
    importances.sort(key=lambda t: t[1], reverse=True)
    return base, importances


# ===========================================================================
# Modelni saqlash / yuklash (JSON)
# ===========================================================================
def _node_to_dict(node):
    if node.feature is None:
        return {"pred": node.pred, "proba": node.proba}
    return {"f": node.feature, "t": node.thresh,
            "l": _node_to_dict(node.left), "r": _node_to_dict(node.right)}


def _dict_to_node(d):
    node = _Node()
    if "pred" in d:
        node.pred = d["pred"]; node.proba = d["proba"]
        return node
    node.feature = d["f"]; node.thresh = d["t"]
    node.left = _dict_to_node(d["l"]); node.right = _dict_to_node(d["r"])
    return node


def save_model(model, path, feature_names=None, standardizer=None, meta=None):
    trees = []
    for tree in model.trees:
        trees.append({"classes": tree.classes_, "root": _node_to_dict(tree.root)})
    obj = {
        "type": "RandomForest",
        "classes": model.classes_,
        "n_trees": model.n_trees,
        "trees": trees,
        "feature_names": feature_names,
        "standardizer": ({"mean": standardizer.mean, "std": standardizer.std}
                         if standardizer else None),
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path


def load_model(path):
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    rf = RandomForest(n_trees=obj.get("n_trees", len(obj["trees"])))
    rf.classes_ = obj["classes"]
    rf.trees = []
    for td in obj["trees"]:
        t = DecisionTree()
        t.classes_ = td["classes"]
        t.root = _dict_to_node(td["root"])
        rf.trees.append(t)
    std = None
    if obj.get("standardizer"):
        std = Standardizer()
        std.mean = obj["standardizer"]["mean"]
        std.std = obj["standardizer"]["std"]
    return {"model": rf, "feature_names": obj.get("feature_names"),
            "standardizer": std, "meta": obj.get("meta", {})}
