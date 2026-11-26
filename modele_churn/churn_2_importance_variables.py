"""modèle churn ClientCo : importance des features (à lancer après churn_1).

Matrice features x indicateurs (resultats/importance/matrice_importance.csv) :
  permutation_ca_capte  baisse du CA à risque capté par la liste si on mélange la feature
  permutation_pr_auc    baisse de la PR-AUC si on mélange la feature
  gain_xgboost          part du gain de la feature dans les arbres
  auc_seule             AUC de la feature utilisée seule
  moyenne_churners / moyenne_restants / ratio
  permutation_total / permutation_partiel   importance par type de churn
  rang_moyen            rang moyen sur les 4 premiers indicateurs

Graphiques : matrice_importance.png, top_variables.png, correlations.png.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import churn_commun as C

ICI = os.path.dirname(os.path.abspath(__file__))


def ca_capte(y, score, perte, ca_ref, K):
    o = np.argsort(-(score * ca_ref))[:K]
    return perte[o].sum() / max(perte.sum(), 1)


def permutation(modele, X, mesure, n_rep=3, seed=0):
    """Baisse moyenne de la mesure quand une feature est mélangée."""
    rng = np.random.default_rng(seed)
    base = mesure(modele.predict_proba(X)[:, 1])
    out = {}
    for v in X.columns:
        pertes = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[v] = rng.permutation(Xp[v].values)
            pertes.append(base - mesure(modele.predict_proba(Xp)[:, 1]))
        out[v] = float(np.mean(pertes))
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ICI, "cache", "photos_mensuelles.pkl"))
    ap.add_argument("--sortie", default=os.path.join(ICI, "resultats", "importance"))
    ap.add_argument("--echantillon", type=int, default=30000, help="clients de test utilisés pour les permutations")
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)
    with open(a.cache, "rb") as f:
        P = pickle.load(f)
    train, test = P["train"], P["test"]
    V = C.VARIABLES

    nom, modele = next(iter(C.boosting(P["seed"]).items()))
    print(f"Modèle analysé : {nom} (cible : départ total ou forte baisse)")
    modele.fit(train[V], train["churn"])

    sub = test.sample(min(len(test), a.echantillon), random_state=0)
    K = max(int(P["K_trim"] * len(sub) / len(test)), 1)
    X = sub[V]
    perte = np.where(sub["churn"] == 1, sub["perte_trim"], 0)
    ref = sub["ca_ref_trim"].values
    M = pd.DataFrame(index=V)
    M["description"] = pd.Series(C.DESCRIPTION_VARIABLES)
    print("Permutations (CA capté)...")
    M["permutation_ca_capte"] = permutation(modele, X, lambda s: ca_capte(sub["churn"].values, s, perte, ref, K))
    print("Permutations (PR-AUC)...")
    M["permutation_pr_auc"] = permutation(modele, X, lambda s: average_precision_score(sub["churn"], s))
    try:
        g = pd.Series(modele.get_booster().get_score(importance_type="gain"))
        M["gain_xgboost"] = (g / g.sum()).reindex(V).fillna(0)
    except AttributeError:
        M["gain_xgboost"] = np.nan
    M["auc_seule"] = [max(roc_auc_score(sub["churn"], X[v].fillna(X[v].median())),
                          1 - roc_auc_score(sub["churn"], X[v].fillna(X[v].median()))) for v in V]
    M["moyenne_churners"] = sub.loc[sub["churn"] == 1, V].mean()
    M["moyenne_restants"] = sub.loc[sub["churn"] == 0, V].mean()
    M["ratio_churners_restants"] = M["moyenne_churners"] / M["moyenne_restants"].replace(0, np.nan)
    for cible, col in (("churn_total", "permutation_total"), ("churn_partiel", "permutation_partiel")):
        m2 = next(iter(C.boosting(P["seed"]).values()))
        m2.fit(train[V], train[cible])
        print(f"Permutations ({cible})...")
        M[col] = permutation(m2, X, lambda s, c=cible: average_precision_score(sub[c], s), n_rep=2)
    indic = ["permutation_ca_capte", "permutation_pr_auc", "gain_xgboost", "auc_seule"]
    M["rang_moyen"] = M[indic].rank(ascending=False).mean(axis=1)
    M = M.sort_values("rang_moyen")
    M.round(4).to_csv(os.path.join(a.sortie, "matrice_importance.csv"), sep=";", index_label="feature")

    corr = sub[V + ["churn"]].corr(method="spearman")
    corr.round(3).to_csv(os.path.join(a.sortie, "correlations.csv"), sep=";")
    graphiques(M, corr, indic + ["permutation_total", "permutation_partiel"], a.sortie)

    pd.set_option("display.width", 200)
    print("\n== Top 10 (rang moyen) ==")
    print(M[["permutation_ca_capte", "permutation_pr_auc", "gain_xgboost", "auc_seule", "rang_moyen"]].head(10).round(4))
    doublons = [(i, j, round(corr.loc[i, j], 2)) for k, i in enumerate(V) for j in V[k + 1:] if abs(corr.loc[i, j]) > 0.9]
    if doublons:
        print("\nFeatures quasi redondantes (|corrélation| > 0,9) :", doublons)
    print(f"\nTout est dans {os.path.relpath(a.sortie)}")


def graphiques(M, corr, cols, dossier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = M[cols].astype(float)
    N = (N - N.min()) / (N.max() - N.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(N.fillna(0).values, cmap="Greens", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(N))); ax.set_yticklabels(N.index, fontsize=8)
    ax.set_title("Importance des features (0 = la plus faible, 1 = la plus forte, par indicateur)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "matrice_importance.png"), dpi=150); plt.close(fig)

    s = M["permutation_ca_capte"].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index, 100 * s.values, color="#197A56")
    ax.set_xlabel("Points de CA à risque capté perdus si la feature est mélangée")
    ax.set_title("Les features qui comptent le plus pour la liste d'appels")
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "top_variables.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Corrélations de Spearman entre features")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "correlations.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
