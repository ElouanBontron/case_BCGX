"""
=====================================================================================================
 Dannyl MOUHEB – Modèle churn ClientCo – SCRIPT 3 : SCORER TOUS LES MOIS OU TOUTES LES 2 SEMAINES ?
=====================================================================================================

    python dannyl_modele_churn/dannyl_3_comparaison_mensuel_bimensuel.py

-----------------------------------------------------------------------------------------------------
 LA QUESTION
-----------------------------------------------------------------------------------------------------
 Notre modèle produit une liste d'appels le 1er de chaque mois ; celui d'Elouan note les clients
 chaque lundi. Faut-il recalculer la liste plus souvent ? Plus souvent = information plus fraîche
 (on repère un client plus tôt dans son silence), mais des listes plus petites à chaque fois.

-----------------------------------------------------------------------------------------------------
 LA MÉTHODE : on rejoue le processus réel, à capacité commerciale ÉGALE
-----------------------------------------------------------------------------------------------------
 - Clients : 80 % pour apprendre, 20 % pour évaluer (les mêmes que le script 1, même graine).
 - Apprentissage : photos du 1er octobre 2023 au 15 janvier 2024, à la fréquence testée.
 - Fenêtre de test : du 1er février au ~25 mars 2024 (dernière date dont la cible est observable).
     * MENSUEL     : listes le 1er février et le 1er mars, de k clients chacune ;
     * BIMENSUEL   : listes le 1er et le 15 de chaque mois, de k/2 clients chacune ;
     * BIMENSUEL (modèle mensuel) : même calendrier que bimensuel mais avec le modèle appris en mensuel,
       pour séparer l'effet « scorer plus souvent » de l'effet « apprendre sur plus de photos ».
   Un client contacté ne l'est pas une deuxième fois dans la fenêtre.
 - Pour chaque appel, on regarde si le client était vraiment en train de partir et le CA à risque.

-----------------------------------------------------------------------------------------------------
 LES INDICATEURS COMPARÉS
-----------------------------------------------------------------------------------------------------
 vrais_departs / precision        combien de clients appelés partaient vraiment
 part_perte_captee                part du CA à risque de la fenêtre que les appels ont atteinte
 marge_max_an_groupe_meur         marge annuelle maximale (si tout ce CA était récupéré), groupe
 silence_median_au_contact_j      silence du client au moment de l'appel : plus c'est court, plus on
 silence_relatif_median           l'appelle tôt, plus on a de chances de le retenir (non chiffré ici)

 Sorties : dannyl_modele_churn/resultats/frequences/comparaison_mensuel_bimensuel.csv / .json / .png
"""

import argparse
import json
import os

import pandas as pd

import dannyl_commun as C

ICI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chemin", nargs="?", default=os.path.join("data", "dataset_sales_2026.csv"))
    ap.add_argument("--cache", default=os.path.join(ICI, "cache", "agregats.pkl"))
    ap.add_argument("--sortie", default=os.path.join(ICI, "resultats", "frequences"))
    ap.add_argument("--contacts-par-mois", type=float, default=20)
    ap.add_argument("--part-clients-test", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)

    agg = C.agreger(a.chemin, a.cache)
    rep = C.representativite(agg)
    part = rep["part_ca_groupe_2023"] or rep["part_ca_groupe"]
    D = C.preparer(agg)
    photos = C.Photos(D)
    g = C.separer_clients(D, {"train": 1 - a.part_clients_test, "test": a.part_clients_test}, a.seed)
    debut_train, fin_train, debut_test, dernier_test = C.fenetres(D)
    print(f"Apprentissage {debut_train.date()} -> {fin_train.date()} | test {debut_test.date()} -> {dernier_test.date()}")

    plans = {
        "Mensuel (1er du mois)": (C.dates_regulieres(debut_train, fin_train, mensuel=True),
                                  C.dates_regulieres(debut_test, dernier_test, mensuel=True), 1.0),
        "Bimensuel (1er et 15)": (C.dates_bimensuelles(debut_train, fin_train),
                                  C.dates_bimensuelles(debut_test, dernier_test), 0.5),
    }
    lignes, res, modele_mensuel = [], {}, None
    for nom, (d_train, d_test, pas) in plans.items():
        print(f"\n== {nom} : {len(d_train)} photos d'apprentissage, {len(d_test)} listes de test ==")
        r = C.evaluer_frequence(photos, d_train, d_test, g["train"], g["test"], pas, part,
                                contacts_par_mois=a.contacts_par_mois, seed=a.seed)
        if nom.startswith("Mensuel"):
            modele_mensuel = r["modele"]
        res[nom] = r
    # effet pur de la fréquence de scoring : bimensuel avec le modèle appris en mensuel
    d_test, pas = plans["Bimensuel (1er et 15)"][1], 0.5
    print("\n== Bimensuel, modèle appris en mensuel ==")
    res["Bimensuel (modèle mensuel)"] = C.evaluer_frequence(photos, [], d_test, g["train"], g["test"], pas, part,
                                                            contacts_par_mois=a.contacts_par_mois, seed=a.seed,
                                                            modele=modele_mensuel)
    for nom, r in res.items():
        for methode in (r["nom_boosting"], C.REGLE):
            lignes.append(dict(frequence=nom, methode=methode, photos_apprentissage=r["n_photos_train"],
                               listes_test=r["n_dates_test"], clients_par_liste=r["k_par_date"], **r[methode]))
    tab = pd.DataFrame(lignes)
    tab.to_csv(os.path.join(a.sortie, "comparaison_mensuel_bimensuel.csv"), sep=";", index=False)
    with open(os.path.join(a.sortie, "comparaison_mensuel_bimensuel.json"), "w", encoding="utf-8") as f:
        json.dump(lignes, f, ensure_ascii=False, indent=2, default=float)
    pd.set_option("display.width", 200)
    print("\n", tab[["frequence", "methode", "contacts", "vrais_departs", "precision", "part_perte_captee",
                     "marge_max_an_groupe_meur", "silence_median_au_contact_j"]].to_string(index=False))
    graphique(tab, a.sortie)
    b = tab[tab["methode"] != C.REGLE].set_index("frequence")
    meilleur = b["marge_max_an_groupe_meur"].idxmax()
    print(f"\nPlus de marge atteignable : {meilleur}. À comparer avec le coût opérationnel "
          f"(listes plus fréquentes = plus de coordination pour les commerciaux).")
    print(f"Tout est dans {os.path.abspath(a.sortie)}")


def graphique(tab, dossier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = tab[tab["methode"] != C.REGLE]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, titre in zip(axes, ["marge_max_an_groupe_meur", "precision", "silence_median_au_contact_j"],
                              ["Marge max / an, groupe (M€)", "Précision de la liste", "Silence médian au contact (j)"]):
        ax.bar(range(len(b)), b[col], color=["#197A56", "#29BA74", "#A7D9BF"][:len(b)])
        ax.set_xticks(range(len(b))); ax.set_xticklabels(b["frequence"], rotation=15, ha="right", fontsize=8)
        ax.set_title(titre, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "comparaison_mensuel_bimensuel.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
