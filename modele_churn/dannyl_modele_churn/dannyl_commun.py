"""
=====================================================================================================
 Dannyl MOUHEB – Modèle churn ClientCo – MODULE COMMUN
=====================================================================================================

Ce module contient tout ce qui est partagé par les scripts dannyl_1 à dannyl_4 :

  1. agreger()            lit le CSV de 4 Go par morceaux et le résume (cache réutilisable)
  2. representativite()   mesure la part du groupe couverte par le dataset (pour extrapoler)
  3. preparer()           met les données en mémoire sous une forme rapide à interroger
  4. photo()              construit, pour une date T, une ligne par client éligible avec
                          ses FEATURES (calculées avant T) et ses CIBLES (observées après T)
  5. modeles()            les modèles comparés (règle simple, régression logistique, XGBoost)
  6. evaluer()...         les métriques métier sous contrainte de capacité commerciale
  7. simuler_relances()   rejoue le processus réel (liste d'appels à chaque date de scoring)

-----------------------------------------------------------------------------------------------------
 VOCABULAIRE
-----------------------------------------------------------------------------------------------------
 - Photo (snapshot) à la date T : l'état de chaque client vu le jour T au matin. Les features
   n'utilisent QUE des achats strictement antérieurs à T ; les cibles n'utilisent QUE des achats
   à partir de T. C'est ce qui empêche le modèle de « tricher » en voyant le futur.
 - Feature (variable) : une information sur le client que le modèle utilise pour deviner s'il va
   partir (ex. : nombre de jours depuis son dernier achat).
 - Cible (label) : ce qu'on cherche à prédire, observé après T (le client est-il parti ?).
 - Jour d'achat : un jour où le CA net du client est > 0. Un jour avec uniquement des retours
   (CA net ≤ 0) n'est PAS un jour d'achat (il compte toutefois dans le CA net).

-----------------------------------------------------------------------------------------------------
 CLIENTS ÉLIGIBLES À LA DATE T (ceux que le modèle note)
-----------------------------------------------------------------------------------------------------
 - au moins 3 jours d'achat sur les 365 jours avant T  -> on exclut les clients occasionnels ;
 - un rythme d'achat connu (délai médian entre deux jours d'achat calculable) ;
 - pas déjà partis : silence actuel ≤ seuil de départ (défini ci-dessous).

-----------------------------------------------------------------------------------------------------
 CIBLES (observées après T)
-----------------------------------------------------------------------------------------------------
 seuil = 3 × délai médian entre deux jours d'achat, borné entre 90 et 180 jours.
   (plancher 90 j : sinon un client qui achète tous les 2 jours serait « perdu » après 6 jours ;
    plafond 180 j : il faut pouvoir observer la cible avant la fin des données)
 - churn_total   = 1 si aucun jour d'achat dans [T, T + seuil[
 - churn_partiel = 1 si pas de départ total ET CA net de [T, T+91 j[ < 50 % de la référence,
                   référence = min(CA du même trimestre l'an dernier, CA moyen par trimestre sur 12 mois)
                   (le « même trimestre l'an dernier » neutralise la saisonnalité)
 - churn         = churn_total OU churn_partiel   -> CIBLE PRINCIPALE du modèle
 - ca_ref_trim   = CA trimestriel de référence = moyenne(même trimestre N-1, trimestre moyen 12 mois)
 - perte_trim    = CA à risque = max(ca_ref_trim - CA réalisé sur [T, T+91 j[, 0) pour un churner, 0 sinon

-----------------------------------------------------------------------------------------------------
 FEATURES (toutes calculées avec des données antérieures à T) – voir DESCRIPTION_VARIABLES
-----------------------------------------------------------------------------------------------------
 Fenêtres en jours glissants : 1m = 30 j, 3m = 91 j, 6m = 182 j, 12m = 365 j.
 Les features « mensuelles » (produits, canaux, retours, mois actifs) utilisent les mois calendaires
 complets avant le mois de T (si T tombe le 15, le mois en cours n'est pas encore complet : on
 l'ignore, soit un décalage d'au plus un mois sur ces features lentes).
"""

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------- constantes
CANAUX = {
    "by phone": "tel",
    "at the store": "agence",
    "online": "web",
    "during the visit of a sales rep": "visite",
    "other": "autre",
}
CA_GROUPE_2023 = 15504.0      # M€ – Financials.xlsx, ClientCo 2023
MARGE_BRUTE = 0.247           # 3 822 / 15 504 – Financials.xlsx 2023
NB_AGENCES_GROUPE = 1000      # « > 1 000 agences » (Facts ClientCo, SWOT)
NB_CLIENTS_GROUPE = 450000    # « > 450 k clients actifs » (Facts ClientCo, SWOT)
COUT_UC_12_MOIS = 0.75        # M€ – 60 ETP-mois × 150 k€ / 12 (slide BCG « FTE needs »)
JOURS_PAR_MOIS = 30.44
HORIZON_MAX = 180             # plafond du seuil de départ total (jours)

DESCRIPTION_VARIABLES = {
    # --- montants (CA net, retours déduits)
    "ca_1m": "CA net des 30 derniers jours",
    "ca_3m": "CA net des 91 derniers jours",
    "ca_6m": "CA net des 182 derniers jours",
    "ca_12m": "CA net des 365 derniers jours",
    "panier_moyen": "CA 12 mois / nombre de jours d'achat sur 12 mois",
    # --- fréquence
    "jours_1m": "nombre de jours d'achat sur 30 jours",
    "jours_3m": "nombre de jours d'achat sur 91 jours",
    "jours_6m": "nombre de jours d'achat sur 182 jours",
    "jours_12m": "nombre de jours d'achat sur 365 jours",
    "mois_actifs_12m": "nombre de mois calendaires avec au moins un achat (12 derniers mois complets)",
    # --- tendances (bornées entre 0 et 10 pour éviter les ratios absurdes quand la base est ~0)
    "tendance_3m": "CA des 91 derniers jours / CA des 91 jours précédents",
    "tendance_6m": "CA des 182 derniers jours / CA des 182 jours précédents",
    # --- saisonnalité propre au client
    "ca_meme_trim_n1": "CA du trimestre à venir, l'an dernier ([T-365 j, T-274 j[)",
    "indice_saison": "CA du même trimestre N-1 / CA trimestriel moyen sur 12 mois",
    # --- rythme d'achat
    "delai_median": "délai médian (jours) entre deux jours d'achat, sur 18 mois",
    "regularite_cv": "écart-type / moyenne des délais (faible = client régulier)",
    "recence_j": "jours depuis le dernier jour d'achat",
    "recence_ratio": "silence actuel / délai médian habituel (> 1 = en retard sur son rythme)",
    "anciennete_j": "jours depuis le premier achat connu (tronqué au début des données)",
    # --- comportement
    "produits_mois_moy": "nombre moyen de produits différents achetés par mois actif",
    "produits_3m": "somme des produits différents achetés par mois, 3 derniers mois complets",
    "part_retours": "retours / (CA net + retours) sur 12 mois",
    "part_tel": "part du CA par téléphone (12 mois)",
    "part_agence": "part du CA en agence (12 mois)",
    "part_web": "part du CA en ligne (12 mois)",
    "part_visite": "part du CA lors d'une visite commerciale (12 mois)",
    "part_autre": "part du CA par un autre canal (12 mois)",
    # --- contexte
    "taille_agence": "CA total de l'agence principale du client (attribut fixe sur la période)",
}
VARIABLES = list(DESCRIPTION_VARIABLES)


# ================================================================================ 1. AGRÉGATION
def agreger(chemin, cache, taille_morceau=2_000_000):
    """Lit le CSV par morceaux de 2 M lignes et produit 3 tables + un diagnostic :
       cm : client x mois  -> CA net, retours, lignes, jours, produits distincts, CA par canal
       cd : client x jour  -> CA net du jour (retours compris)
       cb : client x agence -> CA net (pour trouver l'agence principale)
    Le résultat est sauvegardé (pickle) : les lancements suivants prennent quelques secondes.
    Les doublons exacts sont retirés morceau par morceau."""
    if os.path.exists(cache):
        print(f"Cache trouvé : {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    cm_parts, cd_parts, cb_parts = [], [], []
    produits = set()
    diag = dict(lignes=0, doublons=0, lignes_neg=0, ca_neg=0.0, lignes_nulles=0)
    lecteur = pd.read_csv(
        chemin, sep=";", chunksize=taille_morceau,
        dtype={"product_id": "int64", "client_id": "int64", "sales_net": "float64",
               "quantity": "float64", "order_channel": "category", "branch_id": "int64"},
    )
    for i, m in enumerate(lecteur):
        n0 = len(m)
        m = m.drop_duplicates()
        diag["lignes"] += n0
        diag["doublons"] += n0 - len(m)
        m["date_order"] = pd.to_datetime(m["date_order"], errors="coerce").dt.normalize()
        m = m.dropna(subset=["date_order"])
        diag["lignes_neg"] += int((m["sales_net"] < 0).sum())
        diag["ca_neg"] += float(m.loc[m["sales_net"] < 0, "sales_net"].sum())
        diag["lignes_nulles"] += int((m["sales_net"] == 0).sum())
        produits.update(m["product_id"].unique().tolist())
        m["mois"] = m["date_order"].dt.to_period("M")
        m["canal"] = m["order_channel"].astype(str).map(CANAUX).fillna("autre")
        m["ret"] = (-m["sales_net"]).clip(lower=0)
        g = m.groupby(["client_id", "mois"])
        cm = pd.DataFrame({
            "ca": g["sales_net"].sum(), "ret": g["ret"].sum(), "lignes": g.size(),
            "jours": g["date_order"].nunique(), "produits": g["product_id"].nunique(),
        })
        canal = m.pivot_table(index=["client_id", "mois"], columns="canal", values="sales_net",
                              aggfunc="sum", fill_value=0.0)
        canal.columns = ["ca_" + c for c in canal.columns]
        cm_parts.append(cm.join(canal))
        cd_parts.append(m.groupby(["client_id", "date_order"])["sales_net"].sum())
        cb_parts.append(m.groupby(["client_id", "branch_id"])["sales_net"].sum())
        print(f"  morceau {i + 1} : {diag['lignes']:,} lignes lues ({time.time() - t0:.0f} s)", end="\r")
    print()
    cm = pd.concat(cm_parts).groupby(level=[0, 1]).sum().fillna(0.0)
    cd = pd.concat(cd_parts).groupby(level=[0, 1]).sum().reset_index()
    cd.columns = ["client_id", "date", "ca"]
    cb = pd.concat(cb_parts).groupby(level=[0, 1]).sum().reset_index()
    cb.columns = ["client_id", "branch_id", "ca"]
    diag["produits"] = len(produits)
    out = dict(cm=cm, cd=cd, cb=cb, diag=diag)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"Agrégation terminée en {time.time() - t0:.0f} s -> {cache}")
    return out


# ================================================================================ 2. REPRÉSENTATIVITÉ
def representativite(agg):
    """Part du groupe couverte par le dataset. On extrapole par le CA 2023 (année complète) :
    2024 s'arrête en septembre, l'utiliser surestimerait le facteur d'extrapolation."""
    cd, cb, diag = agg["cd"], agg["cb"], agg["diag"]
    fin = cd["date"].max()
    debut12 = fin - pd.DateOffset(months=12)
    ca12 = cd.loc[cd["date"] > debut12, "ca"].sum() / 1e6
    ca23 = cd.loc[cd["date"].dt.year == 2023, "ca"].sum() / 1e6
    clients12 = cd.loc[cd["date"] > debut12, "client_id"].nunique()
    agences = cb["branch_id"].nunique()
    return dict(
        periode_debut=str(cd["date"].min().date()), periode_fin=str(fin.date()),
        lignes=diag["lignes"], doublons=diag["doublons"], lignes_neg=diag["lignes_neg"],
        ca_retours_meur=round(-diag["ca_neg"] / 1e6, 1), lignes_nulles=diag["lignes_nulles"],
        produits=diag["produits"], clients=int(cd["client_id"].nunique()), agences=int(agences),
        ca_dataset_12m_meur=round(ca12, 0), part_ca_groupe=round(ca12 / CA_GROUPE_2023, 3),
        ca_dataset_2023_meur=round(ca23, 0), part_ca_groupe_2023=round(ca23 / CA_GROUPE_2023, 3),
        facteur_groupe=round(CA_GROUPE_2023 / max(ca23, 1e-9), 2),
        part_agences=round(agences / NB_AGENCES_GROUPE, 3),
        clients_actifs_12m=int(clients12), part_clients=round(clients12 / NB_CLIENTS_GROUPE, 3),
    )


# ================================================================================ 3. PRÉPARATION
class Donnees:
    """Données en mémoire sous forme de tableaux numpy indexés par client (accès très rapide)."""


def preparer(agg):
    D = Donnees()
    cd = agg["cd"].sort_values(["client_id", "date"])
    D.clients = np.sort(cd["client_id"].unique())
    idx = pd.Index(D.clients)
    D.n = len(D.clients)
    D.epoch = pd.Timestamp("2000-01-01")
    jour = (cd["date"] - D.epoch).dt.days.values.astype(np.int32)
    D.c = idx.get_indexer(cd["client_id"].values)
    D.d, D.ca = jour, cd["ca"].values.astype(np.float64)
    # jours d'achat (CA net > 0) et délai avec le jour d'achat précédent du même client
    pos = D.ca > 0
    D.pc, D.pd = D.c[pos], D.d[pos]
    gap = np.diff(D.pd, prepend=-1).astype(np.float64)
    nouveau = np.r_[True, D.pc[1:] != D.pc[:-1]]
    gap[nouveau] = np.nan
    D.pgap = gap
    D.premier = np.full(D.n, np.iinfo(np.int32).max, dtype=np.int64)
    np.minimum.at(D.premier, D.pc, D.pd)
    D.debut, D.fin = int(D.d.min()), int(D.d.max())
    # matrices mensuelles alignées sur les clients
    cm = agg["cm"]
    D.mois = pd.period_range(cm.index.get_level_values(1).min(), cm.index.get_level_values(1).max(), freq="M")
    full = cm.reindex(pd.MultiIndex.from_product([D.clients, D.mois]), fill_value=0.0)
    D.M = {k: full[k].values.reshape(D.n, len(D.mois)) for k in full.columns}
    # agence principale (plus gros CA) et taille de cette agence
    cb = agg["cb"]
    princ = cb.sort_values("ca").drop_duplicates("client_id", keep="last").set_index("client_id")["branch_id"]
    taille = cb.groupby("branch_id")["ca"].sum()
    D.taille_agence = princ.map(taille).reindex(D.clients).fillna(0).values
    return D


def jour(D, T):
    return int((pd.Timestamp(T) - D.epoch).days)


def date_de(D, j):
    return D.epoch + pd.Timedelta(days=int(j))


# ================================================================================ 4. PHOTO À LA DATE T
def photo(D, T, clients_idx=None):
    """Une ligne par client éligible à la date T : features (avant T) + cibles (après T).
    clients_idx : sous-ensemble d'indices clients (sinon tous)."""
    t = jour(D, T)
    if t + HORIZON_MAX > D.fin:
        raise ValueError(f"{pd.Timestamp(T).date()} : pas assez de recul pour observer la cible (180 j).")
    n = D.n
    somme = lambda m, w: np.bincount(D.c[m], weights=w[m] if w is not None else None, minlength=n)
    age = t - D.d                                           # âge en jours d'une ligne (>0 = passé)
    f = {}
    for nom, w in (("1m", 30), ("3m", 91), ("6m", 182), ("12m", 365)):
        f[f"ca_{nom}"] = somme((age >= 1) & (age <= w), D.ca)
    ca_3m_prec = somme((age > 91) & (age <= 182), D.ca)
    ca_6m_prec = somme((age > 182) & (age <= 364), D.ca)
    f["ca_meme_trim_n1"] = somme((age > 274) & (age <= 365), D.ca)
    ca_next_q = somme((age <= 0) & (age > -91), D.ca)
    page = t - D.pd
    for nom, w in (("1m", 30), ("3m", 91), ("6m", 182), ("12m", 365)):
        m = (page >= 1) & (page <= w)
        f[f"jours_{nom}"] = np.bincount(D.pc[m], minlength=n).astype(float)
    # rythme d'achat sur 18 mois
    m = (page >= 1) & (page <= 548)
    g = pd.Series(D.pgap[m]).groupby(D.pc[m])
    med, moy, et = g.median(), g.mean(), g.std()
    dernier = pd.Series(D.pd[m]).groupby(D.pc[m]).max()
    mf = (page <= 0) & (page > -HORIZON_MAX - 1)
    prochain = pd.Series(D.pd[mf]).groupby(D.pc[mf]).min()
    to = lambda s: s.reindex(np.arange(n)).values
    f["delai_median"] = to(med)
    f["regularite_cv"] = to(et) / np.maximum(to(moy), 1e-9)
    f["recence_j"] = t - to(dernier)
    f["anciennete_j"] = np.where(D.premier < t, t - D.premier, np.nan)
    prochain_j = to(prochain) - t
    f["recence_ratio"] = f["recence_j"] / np.maximum(f["delai_median"], 1)
    f["tendance_3m"] = np.clip(f["ca_3m"] / np.maximum(ca_3m_prec, 1), 0, 10)
    f["tendance_6m"] = np.clip(f["ca_6m"] / np.maximum(ca_6m_prec, 1), 0, 10)
    trim_moy = f["ca_12m"] / 4
    f["indice_saison"] = np.clip(f["ca_meme_trim_n1"] / np.maximum(trim_moy, 1), 0, 10)
    f["panier_moyen"] = f["ca_12m"] / np.maximum(f["jours_12m"], 1)
    # features mensuelles : 12 mois calendaires complets avant le mois de T
    tm = D.mois.get_loc(pd.Period(pd.Timestamp(T), "M"))
    sm = lambda k, a: D.M[k][:, max(tm - a, 0):tm].sum(1)
    f["mois_actifs_12m"] = (D.M["jours"][:, max(tm - 12, 0):tm] > 0).sum(1).astype(float)
    f["produits_mois_moy"] = sm("produits", 12) / np.maximum(f["mois_actifs_12m"], 1)
    f["produits_3m"] = sm("produits", 3)
    f["part_retours"] = sm("ret", 12) / np.maximum(sm("ca", 12) + sm("ret", 12), 1)
    canaux = [k for k in D.M if k.startswith("ca_")]
    tot = np.maximum(sum(sm(k, 12) for k in canaux), 1)
    for nom in CANAUX.values():
        k = "ca_" + nom
        f["part_" + nom] = sm(k, 12) / tot if k in D.M else np.zeros(n)
    f["taille_agence"] = D.taille_agence

    df = pd.DataFrame(f)
    df.insert(0, "client_id", D.clients)
    seuil = np.clip(3 * df["delai_median"], 90, HORIZON_MAX)
    df["seuil_j"] = seuil
    elig = (df["jours_12m"] >= 3) & df["delai_median"].notna() & (df["recence_j"] <= seuil)
    if clients_idx is not None:
        sel = np.zeros(n, bool); sel[clients_idx] = True
        elig &= sel
    # cibles
    total = np.isnan(prochain_j) | (prochain_j >= seuil)
    ref = np.minimum(df["ca_meme_trim_n1"], trim_moy)
    partiel = (~total) & (ref > 0) & (ca_next_q < 0.5 * ref)
    df["ca_trim_suivant"] = ca_next_q
    df["churn_total"] = total.astype(int)
    df["churn_partiel"] = partiel.astype(int)
    df["churn"] = (total | partiel).astype(int)
    df["ca_ref_trim"] = 0.5 * (df["ca_meme_trim_n1"] + trim_moy)
    df["perte_trim"] = np.where(df["churn"] == 1, np.maximum(df["ca_ref_trim"] - ca_next_q, 0), 0.0)
    df["T"] = pd.Timestamp(T)
    df["idx"] = np.arange(n)
    return df[elig.values].reset_index(drop=True)


class Photos:
    """Mémoire des photos déjà calculées (une même date sert à plusieurs scripts / fréquences)."""

    def __init__(self, D):
        self.D, self.cache = D, {}

    def __call__(self, T):
        T = pd.Timestamp(T)
        if T not in self.cache:
            df = photo(self.D, T)
            num = df.select_dtypes("float64").columns
            df[num] = df[num].astype("float32")          # divise la mémoire par 2
            self.cache[T] = df
        return self.cache[T]


# ================================================================================ 5. MODÈLES
REGLE = "Règle métier (silence vs rythme habituel)"
LOGIT = "Régression logistique"


def modeles(seed=42, profondeur=5):
    """Trois approches. XGBoost (mêmes réglages que train_churn.py d'Elouan) si installé,
    sinon le gradient boosting de scikit-learn (même famille d'algorithme)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    log1p = FunctionTransformer(lambda X: np.sign(X) * np.log1p(np.abs(X)))
    m = {
        REGLE: None,
        LOGIT: make_pipeline(SimpleImputer(strategy="median"), log1p, StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    m.update(boosting(seed, profondeur))
    return m


def boosting(seed=42, profondeur=5):
    try:
        from xgboost import XGBClassifier
        return {"Gradient boosting (XGBoost)": XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=profondeur,
            learning_rate=0.05, min_child_weight=10, subsample=0.8, colsample_bytree=0.9,
            n_estimators=300, random_state=seed, n_jobs=-1)}
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return {"Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=profondeur, l2_regularization=1.0, random_state=seed)}


def score_regle(df):
    """Règle simple de référence : plus le client est silencieux par rapport à son rythme habituel,
    plus il est à risque ; doublé si son CA des 3 derniers mois a été divisé par 2."""
    s = df["recence_ratio"].fillna(0).values * (1 + (df["tendance_3m"].values < 0.5))
    return s / (s.max() or 1)


def predire(modele, df):
    return score_regle(df) if modele is None else modele.predict_proba(df[VARIABLES])[:, 1]


# ================================================================================ 6. ÉVALUATION
def evaluer(y, score, perte, ca_ref, K):
    """Métriques sur la liste d'appels = les K clients au plus fort CA à risque attendu
    (probabilité × CA de référence). y : cible 0/1 ; perte : CA à risque (0 si pas churner)."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y).astype(int)
    score, perte, ca_ref = (np.asarray(x, dtype=np.float64) for x in (score, perte, ca_ref))
    ordre = np.argsort(-(score * ca_ref))
    top = ordre[:K]
    dans = np.zeros(len(y), bool); dans[top] = True
    gros = ca_ref >= np.quantile(ca_ref, 0.8)                 # 20 % plus gros clients
    tp = int(y[dans].sum())
    ok = 0 < y.mean() < 1
    return dict(
        auc=round(roc_auc_score(y, score), 3) if ok else None,
        pr_auc=round(average_precision_score(y, score), 3) if ok else None,
        brier=round(brier_score_loss(y, np.clip(score, 0, 1)), 4) if ok else None,
        taux_base=round(y.mean(), 3),
        precision_k=round(tp / max(K, 1), 3),
        rappel_k=round(tp / max(y.sum(), 1), 3),
        perte_captee_k=round(perte[top].sum() / max(perte.sum(), 1), 3),
        perte_captee_k_meur=perte[top].sum() / 1e6,
        rappel_k_gros=round(y[dans & gros].sum() / max(y[gros].sum(), 1), 3),
        confusion=dict(vrais_positifs=tp, faux_positifs=int(K - tp),
                       departs_manques=int(y.sum() - tp), vrais_negatifs=int((~dans & (y == 0)).sum())),
    )


def courbe_gain(score, perte, ca_ref, y):
    """% du CA à risque et % des churners captés selon le % de clients contactés."""
    ordre = np.argsort(-(np.asarray(score) * np.asarray(ca_ref)))
    p = np.asarray(perte)[ordre].cumsum() / max(np.asarray(perte).sum(), 1)
    c = np.asarray(y)[ordre].cumsum() / max(np.asarray(y).sum(), 1)
    n = len(ordre)
    return [dict(pct_clients=q, pct_perte=round(100 * p[max(int(n * q / 100) - 1, 0)], 1),
                 pct_churners=round(100 * c[max(int(n * q / 100) - 1, 0)], 1))
            for q in (1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100)]


def calibration(y, score, nb=10):
    """Probabilité prédite moyenne vs taux de churn observé, par décile de score."""
    d = pd.DataFrame({"y": y, "p": np.clip(score, 0, 1)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), nb, labels=False)
    g = d.groupby("bin").agg(proba_predite=("p", "mean"), taux_observe=("y", "mean"), n=("y", "size"))
    return g.round(3).reset_index().to_dict("records")


def intervalle_perte_captee(score, perte, ca_ref, K, n_boot=200, seed=0):
    """Intervalle de confiance à 90 % de la part de CA à risque captée (rééchantillonnage des clients)."""
    rng = np.random.default_rng(seed)
    score, perte, ca_ref = map(np.asarray, (score, perte, ca_ref))
    n, vals = len(score), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        o = np.argsort(-(score[i] * ca_ref[i]))[:K]
        vals.append(perte[i][o].sum() / max(perte[i].sum(), 1))
    return [round(float(np.quantile(vals, 0.05)), 3), round(float(np.quantile(vals, 0.95)), 3)]


def separer_clients(D, parts, seed=42):
    """Répartit aléatoirement les clients en groupes (ex. {'train': .8, 'test': .2}).
    Un client n'appartient qu'à un groupe : le modèle est testé sur des clients jamais vus."""
    rng = np.random.default_rng(seed)
    ordre = rng.permutation(D.n)
    out, debut = {}, 0
    noms = list(parts)
    for i, nom in enumerate(noms):
        fin = D.n if i == len(noms) - 1 else debut + int(round(parts[nom] * D.n))
        out[nom] = np.sort(ordre[debut:fin])
        debut = fin
    return out


# ================================================================================ 7. SIMULATION OPÉRATIONNELLE
def simuler_relances(modele, photos, dates, clients_idx, k_par_date):
    """Rejoue le processus réel sur une fenêtre de test :
       à chaque date de scoring, on note les clients éligibles (du groupe clients_idx), on retire ceux
       déjà contactés dans la fenêtre, et on contacte les k_par_date plus forts CA à risque attendus.
    Renvoie les contacts (une ligne par appel) et la perte de référence de la fenêtre.
    Sert à comparer des fréquences de scoring à capacité commerciale ÉGALE sur la fenêtre."""
    sel = np.zeros(photos.D.n, bool); sel[clients_idx] = True
    contactes, lignes, pertes_max = set(), [], {}
    for T in dates:
        df = photos(T)
        df = df[sel[df["idx"].values]]
        for i, p in zip(df["idx"].values, df["perte_trim"].values):   # CA à risque de chaque client churner
            if p > pertes_max.get(i, 0):
                pertes_max[i] = p
        df = df[~df["idx"].isin(contactes)].copy()
        df["score"] = predire(modele, df)
        top = df.assign(attendu=df["score"] * df["ca_ref_trim"]).nlargest(k_par_date, "attendu")
        contactes.update(top["idx"].tolist())
        lignes.append(top[["idx", "T", "score", "churn", "churn_total", "churn_partiel", "perte_trim",
                           "recence_j", "delai_median", "ca_ref_trim"]])
    contacts = pd.concat(lignes, ignore_index=True) if lignes else pd.DataFrame()
    return contacts, float(sum(pertes_max.values()))


def resumer_simulation(contacts, perte_fenetre, mois_fenetre, part_clients, part_groupe):
    """Indicateurs d'une simulation, et valeur annualisée à l'échelle du groupe."""
    ch = contacts[contacts["churn"] == 1]
    capte = ch["perte_trim"].sum()
    ann = capte / 1e6 * (12 / mois_fenetre) / part_clients / part_groupe       # M€ CA / an, groupe
    return dict(
        contacts=int(len(contacts)), vrais_departs=int(len(ch)),
        precision=round(len(ch) / max(len(contacts), 1), 3),
        part_perte_captee=round(capte / max(perte_fenetre, 1), 3),
        silence_median_au_contact_j=float(ch["recence_j"].median()) if len(ch) else None,
        silence_relatif_median=float((ch["recence_j"] / ch["delai_median"].clip(lower=1)).median()) if len(ch) else None,
        ca_risque_contacte_an_groupe_meur=round(ann, 0),
        marge_max_an_groupe_meur=round(ann * MARGE_BRUTE, 1),
    )


def dates_regulieres(debut, fin, pas_jours=None, mensuel=False):
    """Dates de scoring : le 1er de chaque mois (mensuel) ou tous les `pas_jours` jours."""
    debut, fin = pd.Timestamp(debut), pd.Timestamp(fin)
    if mensuel:
        return list(pd.date_range(debut, fin, freq="MS"))
    return list(pd.date_range(debut, fin, freq=f"{int(pas_jours)}D"))


def dates_bimensuelles(debut, fin):
    """Le 1er et le 15 de chaque mois (« toutes les 2 semaines » calendaire)."""
    d = pd.date_range(pd.Timestamp(debut) - pd.offsets.MonthBegin(1), pd.Timestamp(fin), freq="MS")
    out = sorted(set(list(d) + [x + pd.Timedelta(days=14) for x in d]))
    return [x for x in out if pd.Timestamp(debut) <= x <= pd.Timestamp(fin)]


def entrainer(modele, photos, dates, clients_idx, cible="churn"):
    """Empile les photos d'apprentissage (clients du groupe d'apprentissage uniquement) et entraîne."""
    sel = np.zeros(photos.D.n, bool); sel[clients_idx] = True
    train = pd.concat([photos(T) for T in dates], ignore_index=True)
    train = train[sel[train["idx"].values]]
    if modele is not None:
        modele.fit(train[VARIABLES], train[cible])
    return modele, train


# ================================================================================ 8. COMPARER DES FRÉQUENCES
def evaluer_frequence(photos, dates_train, dates_test, groupe_train, groupe_eval, pas_mois,
                      part_groupe, reps=200, contacts_par_mois=20, seed=42, profondeur=5, modele=None):
    """Évalue UNE fréquence de scoring, à capacité commerciale égale :
       - apprentissage sur les photos dates_train (clients groupe_train) – sauf si `modele` est fourni ;
       - simulation des relances sur dates_test (clients groupe_eval) : à chaque date, on contacte
         k = 200 × part du groupe × part des clients évalués × 20 contacts/mois × pas (en mois) clients.
         Ainsi, mensuel et bimensuel font le MÊME nombre total d'appels sur la fenêtre.
    Renvoie les indicateurs pour XGBoost et pour la règle simple."""
    D = photos.D
    nom, m = next(iter(boosting(seed, profondeur).items()))
    n_lignes = None
    if modele is None:
        m, train = entrainer(m, photos, dates_train, groupe_train)
        n_lignes = int(len(train))
    else:
        m = modele
    part_eval = len(groupe_eval) / D.n
    k = max(int(round(reps * part_groupe * part_eval * contacts_par_mois * pas_mois)), 1)
    mois_fenetre = len(dates_test) * pas_mois
    out = {"modele": m, "k_par_date": k, "n_dates_test": len(dates_test), "mois_fenetre": mois_fenetre,
           "n_photos_train": len(dates_train), "n_lignes_train": n_lignes}
    for lib, mod in ((nom, m), (REGLE, None)):
        contacts, perte_fenetre = simuler_relances(mod, photos, dates_test, groupe_eval, k)
        out[lib] = resumer_simulation(contacts, perte_fenetre, mois_fenetre, part_eval, part_groupe)
    out["nom_boosting"] = nom
    return out


def fenetres(D):
    """Fenêtres communes aux scripts 3 et 4 :
       apprentissage du 1er octobre 2023 au 15 janvier 2024 ; test du 1er février 2024 jusqu'à la
       dernière date dont la cible est observable (fin des données - 180 jours)."""
    fin = date_de(D, D.fin)
    debut_train = (date_de(D, D.debut) + pd.Timedelta(days=365)).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    dernier_test = fin - pd.Timedelta(days=HORIZON_MAX)
    debut_test = dernier_test.to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
    fin_train = debut_test - pd.Timedelta(days=17)
    return debut_train, fin_train, debut_test, dernier_test
