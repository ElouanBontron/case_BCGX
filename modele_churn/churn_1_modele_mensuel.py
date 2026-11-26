"""modèle churn ClientCo : modèle mensuel (chiffres de la présentation).

python modele_churn/churn_1_modele_mensuel.py [--contacts-par-mois 20]

- Photos le 1er de chaque mois, d'octobre 2023 à mars 2024.
- Clients : 80 % apprentissage, 20 % test.
- Apprentissage : photos d'octobre 2023 à janvier 2024. Test : photo du 1er mars 2024 (février sert de tampon).
- 3 modèles (règle simple, régression logistique, XGBoost) x 3 cibles (total, partiel, les deux).
- Liste d'appels : clients classés par probabilité x CA de référence, coupés à la capacité sur un trimestre.

Hypothèses : 200 commerciaux, 20 contacts par commercial et par mois, part du groupe couverte = CA 2023
du dataset / 15 504 M€, valeur annuelle = 4 x un trimestre, marge brute 24,7 %, coût du use case 0,75 M€/an.
Aucun taux de récupération n'est supposé (sensibilité 10 / 25 / 50 %, point mort).

Sorties dans resultats/mensuel/ : resultats.json, metriques.csv, courbe_gain.png, calibration.csv,
liste_appels_exemple.csv. Les photos sont gardées dans cache/photos_mensuelles.pkl pour le script 2.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

import churn_commun as C

ICI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chemin", nargs="?", default=os.path.join("data", "dataset_sales_2026.csv"))
    ap.add_argument("--cache", default=os.path.join(ICI, "cache", "agregats.pkl"))
    ap.add_argument("--sortie", default=os.path.join(ICI, "resultats", "mensuel"))
    ap.add_argument("--reps", type=float, default=200)
    ap.add_argument("--contacts-par-mois", type=float, default=20)
    ap.add_argument("--part-clients-test", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)
    R = {}

    # --- données et représentativité
    agg = C.agreger(a.chemin, a.cache)
    rep = C.representativite(agg)
    part = rep["part_ca_groupe_2023"] or rep["part_ca_groupe"]
    R["dataset"] = rep
    print("\n== Représentativité ==")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    D = C.preparer(agg)
    photos = C.Photos(D)

    # --- photos, découpages
    fin = C.date_de(D, D.fin)
    T_min = (C.date_de(D, D.debut) + pd.Timedelta(days=365)).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    T_max = (fin - pd.Timedelta(days=C.HORIZON_MAX)).to_period("M").to_timestamp()
    snaps = list(pd.date_range(T_min, T_max, freq="MS"))
    if len(snaps) < 3:
        raise SystemExit("Pas assez d'historique pour au moins 3 photos mensuelles.")
    groupes = C.separer_clients(D, {"train": 1 - a.part_clients_test, "test": a.part_clients_test}, a.seed)
    print(f"\nPhotos : {[d.strftime('%Y-%m') for d in snaps]} | apprentissage <= {snaps[-3]:%Y-%m}, test = {snaps[-1]:%Y-%m}")
    sel_test = np.zeros(D.n, bool); sel_test[groupes["test"]] = True
    test = photos(snaps[-1])
    test = test[sel_test[test["idx"].values]].copy()
    photo_complete = photos(snaps[-1])                   # tous les clients, pour les statistiques

    # capacité : commerciaux couvrant le dataset, rapportée aux clients de test
    reps_ds = a.reps * part
    K_mois_ds = int(round(reps_ds * a.contacts_par_mois))
    K_trim = 3 * max(int(round(K_mois_ds * a.part_clients_test)), 1)
    R["capacite"] = dict(reps_groupe=a.reps, part_groupe=part, reps_dataset=round(reps_ds, 1),
                         contacts_par_mois=a.contacts_par_mois, liste_mensuelle=K_mois_ds,
                         liste_trimestre=3 * K_mois_ds, liste_trimestre_test=K_trim,
                         part_clients_test=a.part_clients_test, contacts_an_groupe=int(a.reps * a.contacts_par_mois * 12),
                         n_clients_test=int(len(test)),
                         part_clients_eligibles_contactes_trim=round(K_trim / len(test), 3))

    # --- descriptif du churn (photo complète)
    desc = {}
    for cible in ("churn_total", "churn_partiel", "churn"):
        s = photo_complete[cible] == 1
        desc[cible] = dict(
            pct_clients=round(s.mean(), 3), nb_clients=int(s.sum()),
            ca_ref_moyen_trim_keur=round(photo_complete.loc[s, "ca_ref_trim"].mean() / 1e3, 1),
            perte_trim_meur=round(photo_complete.loc[s, "perte_trim"].sum() / 1e6, 2),
            part_ca_ref=round(photo_complete.loc[s, "perte_trim"].sum() / max(photo_complete["ca_ref_trim"].sum(), 1), 3),
            ca_risque_an_groupe_meur=round(4 * photo_complete.loc[s, "perte_trim"].sum() / 1e6 / part, 0),
        )
    R["churn_descriptif"] = desc
    lab = ["< 1x", "1-2x", "2-3x", "3-5x", "> 5x"]
    R["silence_vs_rythme"] = (pd.cut(photo_complete["recence_ratio"], [-1, 1, 2, 3, 5, np.inf], labels=lab)
                              .value_counts(normalize=True).reindex(lab).round(3).to_dict())

    # --- entraînement et évaluation
    res, scores, modeles_fit = {}, {}, {}
    dates_train = snaps[:-2]
    for cible in ("churn_total", "churn_partiel", "churn"):
        res[cible] = {}
        for nom, m in C.modeles(a.seed).items():
            m, train = C.entrainer(m, photos, dates_train, groupes["train"], cible)
            s = C.predire(m, test)
            perte = np.where(test[cible] == 1, test["perte_trim"], 0)
            res[cible][nom] = C.evaluer(test[cible], s, perte, test["ca_ref_trim"], K_trim)
            scores[(cible, nom)] = s
            modeles_fit[(cible, nom)] = m
            r = res[cible][nom]
            print(f"  {cible:14s} | {nom:42s} | AUC {r['auc']} | précision {r['precision_k']} | CA à risque capté {r['perte_captee_k']}")
    R["metriques"] = res
    R["taille_apprentissage"] = int(len(train))

    best = max(res["churn"], key=lambda n: res["churn"][n]["perte_captee_k"])
    R["modele_retenu"] = best
    s = scores[("churn", best)]
    test["score"], test["perte_attendue"] = s, s * test["ca_ref_trim"]
    perte = np.where(test["churn"] == 1, test["perte_trim"], 0)
    R["courbe_gain"] = C.courbe_gain(s, perte, test["ca_ref_trim"], test["churn"])
    R["intervalle_90_perte_captee"] = C.intervalle_perte_captee(s, perte, test["ca_ref_trim"], K_trim)
    if best != C.REGLE:
        R["calibration"] = C.calibration(test["churn"].values, s)
        pd.DataFrame(R["calibration"]).to_csv(os.path.join(a.sortie, "calibration.csv"), sep=";", index=False)

    r = res["churn"][best]
    top = test.nlargest(K_trim, "perte_attendue")
    R["lecture"] = dict(
        sur_100_signales_partent=int(round(100 * r["precision_k"])),
        sur_10_churners_detectes=round(10 * r["rappel_k"], 1),
        part_perte_captee=r["perte_captee_k"], rappel_gros_clients=r["rappel_k_gros"],
        part_total_dans_liste=round(top["churn_total"].sum() / max(top["churn"].sum(), 1), 3),
        part_partiel_dans_liste=round(top["churn_partiel"].sum() / max(top["churn"].sum(), 1), 3),
        departs_atteints_an_groupe=int(round(r["precision_k"] * a.reps * a.contacts_par_mois * 12)),
    )

    # --- valorisation (échelle groupe)
    capte_trim = r["perte_captee_k_meur"] / a.part_clients_test
    toute_perte_trim = perte.sum() / 1e6 / a.part_clients_test
    ca_an_capte = 4 * capte_trim / part
    ca_an_tout = 4 * toute_perte_trim / part
    V = dict(ca_a_risque_an_groupe_meur=round(ca_an_tout, 0), ca_a_risque_an_pct_ca=round(ca_an_tout / C.CA_GROUPE_2023, 3),
             ca_a_risque_dans_liste_an_groupe_meur=round(ca_an_capte, 0),
             marge_max_meur=round(ca_an_capte * C.MARGE_BRUTE, 1),
             marge_potentiel_theorique_meur=round(ca_an_tout * C.MARGE_BRUTE, 1))
    for tx in (0.10, 0.25, 0.50):
        V[f"marge_recup_{int(tx * 100)}_meur"] = round(ca_an_capte * C.MARGE_BRUTE * tx, 1)
    for c in (10, 20, 40):                                                # sensibilité à la capacité
        k = 3 * max(int(round(reps_ds * c * a.part_clients_test)), 1)
        o = np.argsort(-(s * test["ca_ref_trim"].values))[:k]
        V[f"marge_max_{c}contacts_meur"] = round(4 * perte[o].sum() / 1e6 / a.part_clients_test / part * C.MARGE_BRUTE, 1)
    V["taux_point_mort"] = round(C.COUT_UC_12_MOIS / max(ca_an_capte * C.MARGE_BRUTE, 1e-9), 4)
    V["taux_implicite_cdo"] = round(0.01 * C.CA_GROUPE_2023 / max(ca_an_capte, 1e-9), 3)
    R["valeur"] = V

    # --- importance rapide (détail : script 2)
    try:
        from sklearn.inspection import permutation_importance
        m = modeles_fit[("churn", best)]
        if m is not None:
            sub = test.sample(min(len(test), 30000), random_state=0)
            pi = permutation_importance(m, sub[C.VARIABLES], sub["churn"], n_repeats=3, random_state=0,
                                        scoring="average_precision")
            R["importance_variables"] = (pd.Series(pi.importances_mean, index=C.VARIABLES)
                                         .sort_values(ascending=False).head(10).round(4).to_dict())
    except Exception as e:
        print("importance non calculée :", e)

    # --- liste d'appels type
    K_mois_test = K_trim // 3
    liste = test.nlargest(K_mois_test, "perte_attendue")[
        ["client_id", "score", "ca_ref_trim", "ca_12m", "perte_attendue", "recence_j", "delai_median", "tendance_3m", "churn"]].copy()
    q = liste["perte_attendue"].rank(pct=True)
    liste["action"] = np.where(q > 0.9, "Visite", "Appel")   # 2 visites + 18 appels par commercial et par mois
    liste.rename(columns={"churn": "parti_en_realite"}).to_csv(os.path.join(a.sortie, "liste_appels_exemple.csv"), index=False, sep=";")
    R["actions"] = liste["action"].value_counts().to_dict()

    # --- sorties
    pd.concat({c: pd.DataFrame({n: {k: v for k, v in d.items() if k != "confusion"} for n, d in v.items()}).T
               for c, v in res.items()}).to_csv(os.path.join(a.sortie, "metriques.csv"), sep=";")
    graphique(R, a.sortie)
    R["parametres"] = dict(ca_groupe_meur=C.CA_GROUPE_2023, marge_brute=C.MARGE_BRUTE, cout_uc_meur=C.COUT_UC_12_MOIS,
                           part_clients_test=a.part_clients_test, n_clients_test=int(len(test)),
                           photos=[d.strftime("%Y-%m") for d in snaps], photos_apprentissage=[d.strftime("%Y-%m") for d in dates_train],
                           test=snaps[-1].strftime("%Y-%m"), variables=C.VARIABLES)
    with open(os.path.join(a.sortie, "resultats.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, default=float)
    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    with open(os.path.join(os.path.dirname(a.cache), "photos_mensuelles.pkl"), "wb") as f:
        pickle.dump(dict(train=train, test=test, modele_retenu=best, K_trim=K_trim, seed=a.seed), f)
    print("\n== Résultats clés ==")
    print(json.dumps({k: R[k] for k in ("capacite", "lecture", "valeur", "intervalle_90_perte_captee")},
                     ensure_ascii=False, indent=2, default=float))
    print(f"\nTout est dans {os.path.relpath(a.sortie)}")


def graphique(R, dossier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = pd.DataFrame(R["courbe_gain"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0] + list(g["pct_clients"]), [0] + list(g["pct_perte"]), marker="o", color="#197A56", label="CA à risque capté")
    ax.plot([0] + list(g["pct_clients"]), [0] + list(g["pct_churners"]), marker="o", color="#B9BDBA", label="Churners détectés")
    ax.plot([0, 100], [0, 100], ls="--", color="#999", label="Au hasard (x % des clients → x % du CA)")
    ax.axvline(100 * R["capacite"]["part_clients_eligibles_contactes_trim"], color="#29BA74", ls=":", label="Capacité")
    ax.set_xlabel("% des clients réguliers contactés (classés par CA à risque attendu)"); ax.set_ylabel("% capté")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    # 2e abscisse : le même % exprimé en clients contactés par mois à l'échelle du groupe
    cap = R["capacite"]
    par_pct = cap["n_clients_test"] / cap["part_clients_test"] / cap["part_groupe"] / 3 / 100
    haut = ax.secondary_xaxis("top", functions=(lambda x: x * par_pct, lambda n: n / par_pct))
    haut.set_xlabel("Clients contactés par mois, à l'échelle du groupe", color="#29BA74")
    haut.tick_params(axis="x", colors="#29BA74")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "courbe_gain.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
