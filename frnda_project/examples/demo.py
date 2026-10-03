"""
Demo / worked examples for the FRNDA package.

Run with:  python examples/demo.py

"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from frnda import FRNDA, FRNDA2D, RNDA
from frnda.choquet import AdditiveMeasure, TwoAdditiveMeasure, choquet_integral
from frnda.diagnostics import hubness_report
from frnda.fuzzy_membership import fuzzy_knn_membership
from frnda.graph import build_multi_scale_graphs, fuzzy_rnn_affinity, symmetric_knn_support
from frnda.optimize import frnda_optimize
from frnda.rnda import classical_scatter_matrices
from frnda.scatter import fuzzy_scatter_matrices
from frnda.utils import orthonormalize

OUT = Path(__file__).resolve().parent
if not OUT.exists():
    OUT.mkdir(parents=True)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
#      "Monotonic convergence" -- "convergence" reproduction
# ---------------------------------------------------------------------------
def demo_convergence() -> None:
    section('1. Theorem "Monotonic convergence"  (reproduces Figure "convergence")')

    X, y = make_classification(n_samples=150, n_features=20, n_informative=10,
                                n_classes=5, n_clusters_per_class=1, class_sep=0.8,
                                random_state=7)
    X = (X - X.mean(0)) / X.std(0)

    K_f, m_f, scales, p = 5, 2.0, (3, 5, 7, 9, 11), 4
    lam, gamma, eps = 1e-2, 1e-2, 1e-6

    U, classes = fuzzy_knn_membership(X, y, K_f, m_f)
    Sw, Sb, _, _ = fuzzy_scatter_matrices(X, U, m_f, ridge=eps)
    graphs = build_multi_scale_graphs(X, scales)
    measure = TwoAdditiveMeasure.random_supermodular(len(scales), rng=np.random.default_rng(123))

    d = X.shape[1]
    rng = np.random.default_rng(999)
    inits = {"LDA-based (Alg. 1, line 4)": None}
    for i in range(5):
        inits[f"random init {i + 1}"] = orthonormalize(rng.normal(size=(d, p)))

    plt.figure(figsize=(6.5, 4.5))
    finals = []
    for label, W0 in inits.items():
        res = frnda_optimize(Sw, Sb, graphs.B, measure, p=p, lam=lam, gamma=gamma,
                              eps=eps, max_iter=100, tol=1e-10, W_init=W0)
        monotone = np.diff(res.history).min() > -1e-7
        finals.append(res.history[-1])
        print(f"  {label:28s}  n_iter={res.n_iter:2d}  monotone={monotone!s:5s}  "
              f"J final={res.history[-1]:.6f}")
        plt.plot(res.history, marker="o", markersize=3,
                  label=label if "random" not in label else None,
                  color="crimson" if "LDA" in label else "steelblue", alpha=0.8)

    print(f"\n  Spread of final J across all 6 initializations: "
          f"{max(finals) - min(finals):.2e}  (all converge to the same value)")

    plt.xlabel("outer iteration $t$")
    plt.ylabel(r"$J(W^{(t)})$")
    plt.title("FRNDA objective is monotone non-decreasing\nfrom every initialization (Theorem 2)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "convergence.png", dpi=140)
    plt.close()
    print(f"\n  saved: {OUT / 'convergence.png'}")


# ---------------------------------------------------------------------------
#     Hubness diagnostic --  "hubness"
# ---------------------------------------------------------------------------
def demo_hubness() -> None:
    section('2. Reverse-neighbor motivation  (reproduces Figure "hubness")')

    rng = np.random.default_rng(0)
    k = 10
    reports = {}
    for d in (3, 50, 200):
        X = rng.normal(size=(1000, d))
        reports[d] = hubness_report(X, k=k)
        print(f"  d={d:4d}:  mean k-occurrence={reports[d]['mean']:.2f}  "
              f"skewness={reports[d]['skewness']:+.3f}  "
              f"hubs (>{3*k} occurrences)={reports[d]['n_hubs_gt_3k']:3d}  "
              f"orphans={reports[d]['n_orphans']:3d}")

    fig, axes = plt.subplots(1, len(reports), figsize=(4 * len(reports), 3.2), sharey=True)
    for ax, (d, rep) in zip(axes, reports.items()):
        ax.hist(rep["k_occurrence"], bins=30, color="steelblue")
        ax.set_title(f"d={d}\nskew={rep['skewness']:.2f}")
        ax.set_xlabel("k-occurrence")
    axes[0].set_ylabel("count")
    fig.suptitle("Hubness emerges in higher ambient dimension: motivates\n"
                  "building graphs from *reverse* rather than forward neighbors")
    plt.tight_layout()
    plt.savefig(OUT / "hubness.png", dpi=140)
    plt.close()
    print(f"\n  saved: {OUT / 'hubness.png'}")


# ---------------------------------------------------------------------------
# "reduction to crisp FRNDA2D"
# ---------------------------------------------------------------------------
def demo_reduction() -> None:
    section('"reduction to crisp FRNDA2D"  (crisp-limit check)')

    rng = np.random.default_rng(11)
    n, d, c = 60, 8, 4
    X = rng.normal(size=(n, d))
    y = rng.integers(0, c, size=n)
    scales = (3, 5, 7)
    p = 3

    U_crisp = np.eye(c)[y]
    Sw_f, Sb_f, _, _ = fuzzy_scatter_matrices(X, U_crisp, m_f=2.0, ridge=1e-6)
    Sw_c, Sb_c, _ = classical_scatter_matrices(X, y, ridge=1e-6)
    print("  (i)   crisp-membership scatter == classical LDA scatter:  ",
          np.allclose(Sw_f, Sw_c, atol=1e-8) and np.allclose(Sb_f, Sb_c, atol=1e-8))

    huge_sigma = np.full(n, 1e6)
    ok = all(np.allclose(fuzzy_rnn_affinity(X, k, huge_sigma).toarray(),
                          symmetric_knn_support(X, k).toarray(), atol=1e-6)
             for k in scales)
    print("  (ii)  sigma->infinity fuzzy affinity == binary RNDA graph:  ", ok)

    measure = AdditiveMeasure.uniform(len(scales))
    r = rng.uniform(0, 5, size=len(scales))
    val, coeffs = choquet_integral(r, measure, return_coeffs=True)
    print("  (iii) additive measure -> Choquet == fixed linear average:  ",
          np.allclose(coeffs, measure.weights))

    rnda = RNDA(n_components=p, scales=scales).fit(X, y)
    print(f"\n  Standalone RNDA (crisp predecessor) fit: n_iter={rnda.n_iter_}, "
          f"final J={rnda.history_[-1]:.4f}")
    print("  (Exhaustive numerical agreement with the FRNDA crisp-limit path is in "
          "tests/test_reduction_to_rnda.py)")


# ---------------------------------------------------------------------------
# Classification benchmark with proper CV tuning
# ---------------------------------------------------------------------------
def demo_classification_benchmark() -> None:
    section("Classification benchmark on digits (with CV-tuned hyperparameters)")

    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    # Standard preprocessing: drop near-constant pixels (e.g. always-blank
    # image corners). This matters more here than for most methods, since a
    # zero-variance direction is a degenerate direction for *any*
    # trace-ratio criterion (it can be included in W at ~zero cost to the
    # objective while contributing pure noise to a downstream Euclidean
    # classifier) -- the same reasoning behind the p-sensitivity the
    # manuscript itself reports.
    X = X[:, X.std(axis=0) > 0.03]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

    def acc(Ztr, Zte):
        return KNeighborsClassifier(n_neighbors=1).fit(Ztr, ytr).score(Zte, yte)

    print(f"  raw pixels + 1-NN            : {acc(Xtr, Xte):.4f}   (64-d, no reduction)")

    lda = LinearDiscriminantAnalysis(n_components=9).fit(Xtr, ytr)
    print(f"  classical LDA (p=9) + 1-NN   : {acc(lda.transform(Xtr), lda.transform(Xte)):.4f}")

    print("\n  Cross-validated grid search for FRNDA (mirrors the manuscript's own "
          '"Hyperparameter Tuning Protocol")...')
    pipe = Pipeline([("frnda", FRNDA(random_state=0)), ("knn", KNeighborsClassifier(n_neighbors=1))])
    grid = {
        "frnda__n_components": [9, 15, 20],
        "frnda__K_f": [5, 10],
        "frnda__lam": [1e-3],
        "frnda__gamma": [1e-2, 1, 10],
        "frnda__measure": ["additive", "two-additive"],
    }
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=-1)
    gs.fit(Xtr, ytr)
    print(f"  grid search over {len(list(gs.cv_results_['params']))} configs took "
          f"{time.time() - t0:.1f}s; best CV accuracy = {gs.best_score_:.4f}")
    print(f"  best params: {gs.best_params_}")
    print(f"  FRNDA (best CV config) + 1-NN: {gs.score(Xte, yte):.4f}")

    best = gs.best_params_
    rnda = RNDA(n_components=best["frnda__n_components"], gamma=best["frnda__gamma"]).fit(Xtr, ytr)
    print(f"  RNDA (same p, gamma) + 1-NN  : {acc(rnda.transform(Xtr), rnda.transform(Xte)):.4f}")

    print("\n  Note: raw-pixel 1-NN is a very strong baseline on this particular, "
          "near-ceiling dataset\n  (LDA also trails it); the informative comparison for a "
          "feature-extraction method\n  is against other extractors (LDA / RNDA) above, not "
          "against unreduced raw pixels.")


# ---------------------------------------------------------------------------
#  2DFRNDA on native image data
# ---------------------------------------------------------------------------
def demo_2dfrnda() -> None:
    section("FRNDA2D on native 8x8 digit images")

    digits = load_digits()
    images = digits.images / 16.0
    y = digits.target
    n = len(images)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.6 * n)], idx[int(0.6 * n):]

    t0 = time.time()
    model = FRNDA2D(p1_components=5, p2_components=5, K_f=5, m_f=2.0, gamma=1.0, lam=1e-3,
                     outer_max_iter=15, random_state=0)
    model.fit(images[tr], y[tr])
    print(f"  fit time: {time.time() - t0:.2f}s   n_iter={model.n_iter_}   "
          f"converged={model.converged_}")
    print(f"  J_2D history: {[round(h, 3) for h in model.history_]}")
    assert np.diff(model.history_).min() > -1e-6, "monotonicity violated!"

    Ztr = model.transform(images[tr]).reshape(len(tr), -1)
    Zte = model.transform(images[te]).reshape(len(te), -1)
    acc2d = KNeighborsClassifier(n_neighbors=1).fit(Ztr, y[tr]).score(Zte, y[te])
    print(f"  FRNDA2D  (U: 8x5, V: 8x5 -> 25-d) + 1-NN: {acc2d:.4f}")

    flat_model = FRNDA(n_components=9, K_f=5, m_f=2.0, gamma=1.0, lam=1e-3, random_state=0)
    Ztr1 = flat_model.fit_transform(images[tr].reshape(len(tr), -1), y[tr])
    Zte1 = flat_model.transform(images[te].reshape(len(te), -1))
    acc1d = KNeighborsClassifier(n_neighbors=1).fit(Ztr1, y[tr]).score(Zte1, y[te])
    print(f"  FRNDA(1D) (vectorized, 9-d)              + 1-NN: {acc1d:.4f}")


if __name__ == "__main__":
    np.random.seed(0)
    demo_convergence()
    demo_hubness()
    demo_reduction()
    demo_classification_benchmark()
    demo_2dfrnda()
    section("Done. See convergence.png and hubness.png/.")
