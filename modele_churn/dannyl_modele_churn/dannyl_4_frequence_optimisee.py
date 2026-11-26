"""
=====================================================================================================
 Dannyl MOUHEB – Modèle churn ClientCo – SCRIPT 4 : LA FRÉQUENCE DE SCORING COMME HYPERPARAMÈTRE
=====================================================================================================

    python dannyl_modele_churn/dannyl_4_frequence_optimisee.py
    python dannyl_modele_churn/dannyl_4_frequence_optimisee.py --pas 7 14 21 28 --profondeurs 3 5 7

-----------------------------------------------------------------------------------------------------
 PRINCIPE
-----------------------------------------------------------------------------------------------------
 La fréquence à laquelle on recalcule la liste d'appels (tous les 7, 14, 21, 28 jours...) est traitée
 comme un hyperparamètre : on la choisit sur des clients de VALIDATION, puis on mesure le résultat
 final sur des clients de TEST qui n'ont servi ni à apprendre ni à choisir (sinon le score serait
 optimiste). En option, on optimise en même temps la profondeur des arbres XGBoost.

 Découpage des clients (au hasard, graine fixe) : 60 % apprentissage, 20 % validation, 20 % test.
 Découpage du temps : apprentissage du 1er octobre 2023 au 15 janvier 2024 ; listes de relance du
 1er février au ~25 mars 2024. Capacité commerciale identique pour toutes les fréquences
 (k par liste proportionnel au pas), un client n'est contacté qu'une fois dans la fenêtre.

 Critère de choix : marge annuelle maximale atteignable par les appels (échelle groupe), puis la
 précision en cas d'égalité. Le silence au moment du contact est rapporté pour juger de la précocité.

 Sorties : dannyl_modele_churn/resultats/frequences/optimisation_frequence.csv / .json / .png
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
    ap.add_argument("--pas", type=int, nargs="+", default=[7, 14, 21, 28], help="pas de scoring en jours")
    ap.add_argument("--profondeurs", type=int, nargs="+", default=[5], help="profondeurs XGBoost testées")
    ap.add_argument("--contacts-par-mois", type=float, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)

    agg = C.agreger(a.chemin, a.cache)
    rep = C.representativite(agg)
    part = rep["part_ca_groupe_2023"] or rep["part_ca_groupe"]
    D = C.preparer(agg)
    g = C.separer_clients(D, {"train": 0.6, "validation": 0.2, "test": 0.2}, a.seed)
    debut_train, fin_train, debut_test, dernier_test = C.fenetres(D)

    lignes = []
    for pas in a.pas:
        photos = C.Photos(D)                    # une mémoire par fréquence (limite la RAM)
        d_train = C.dates_regulieres(debut_train, fin_train, pas_jours=pas)
        d_test = C.dates_regulieres(debut_test, dernier_test, pas_jours=pas)
        for prof in a.profondeurs:
            print(f"\n== pas {pas} j, profondeur {prof} : {len(d_train)} photos d'apprentissage, {len(d_test)} listes ==")
            rv = C.evaluer_frequence(photos, d_train, d_test, g["train"], g["validation"], pas / C.JOURS_PAR_MOIS,
                                     part, contacts_par_mois=a.contacts_par_mois, seed=a.seed, profondeur=prof)
            rt = C.evaluer_frequence(photos, [], d_test, g["train"], g["test"], pas / C.JOURS_PAR_MOIS,
                                     part, contacts_par_mois=a.contacts_par_mois, seed=a.seed, modele=rv["modele"])
            nb = rv["nom_boosting"]
            for jeu, r in (("validation", rv), ("test", rt)):
                lignes.append(dict(pas_jours=pas, profondeur=prof, jeu=jeu, photos_apprentissage=len(d_train),
                                   listes=len(d_test), clients_par_liste=r["k_par_date"], **r[nb],
                                   regle_marge_max_an_groupe_meur=r[C.REGLE]["marge_max_an_groupe_meur"]))
            v = lignes[-2]
            print(f"   validation : marge max {v['marge_max_an_groupe_meur']} M€/an, précision {v['precision']}, "
                  f"silence au contact {v['silence_median_au_contact_j']} j")
        del photos

    tab = pd.DataFrame(lignes)
    val = tab[tab["jeu"] == "validation"].sort_values(["marge_max_an_groupe_meur", "precision"], ascending=False)
    choix = val.iloc[0]
    final = tab[(tab["jeu"] == "test") & (tab["pas_jours"] == choix["pas_jours"]) & (tab["profondeur"] == choix["profondeur"])].iloc[0]
    tab.to_csv(os.path.join(a.sortie, "optimisation_frequence.csv"), sep=";", index=False)
    with open(os.path.join(a.sortie, "optimisation_frequence.json"), "w", encoding="utf-8") as f:
        json.dump(dict(choix=dict(pas_jours=int(choix["pas_jours"]), profondeur=int(choix["profondeur"])),
                       resultat_test=final.to_dict(), toutes=lignes), f, ensure_ascii=False, indent=2, default=float)
    pd.set_option("display.width", 200)
    print("\n", tab[["pas_jours", "profondeur", "jeu", "contacts", "precision", "part_perte_captee",
                     "marge_max_an_groupe_meur", "silence_median_au_contact_j"]].to_string(index=False))
    print(f"\nFréquence retenue (sur la validation) : tous les {int(choix['pas_jours'])} jours, profondeur {int(choix['profondeur'])}")
    print(f"Résultat sur le test : marge max {final['marge_max_an_groupe_meur']} M€/an, précision {final['precision']}")
    graphique(tab, a.sortie)
    print(f"Tout est dans {os.path.abspath(a.sortie)}")


def graphique(tab, dossier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for jeu, style in (("validation", "o-"), ("test", "s--")):
        t = tab[(tab["jeu"] == jeu)].groupby("pas_jours")[["marge_max_an_groupe_meur", "silence_median_au_contact_j"]].max()
        axes[0].plot(t.index, t["marge_max_an_groupe_meur"], style, label=jeu, color="#197A56" if jeu == "validation" else "#29BA74")
        axes[1].plot(t.index, t["silence_median_au_contact_j"], style, label=jeu, color="#197A56" if jeu == "validation" else "#29BA74")
    axes[0].set_title("Marge max / an, groupe (M€)"); axes[1].set_title("Silence médian au contact (jours)")
    for ax in axes:
        ax.set_xlabel("Pas de scoring (jours)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "optimisation_frequence.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
