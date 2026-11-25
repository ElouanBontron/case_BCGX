"""Vérifie les quotas, l'affectation dans le passé et la comptabilité du gain."""

from datetime import date, timedelta
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from simulation_appels import simuler_appels


LUNDI = date(2024, 1, 8)


def ventes(lignes):
    return pl.DataFrame(
        [(client, boutique, jour, ca, quantite) for client, boutique, jour, ca, quantite in lignes],
        schema={
            "client_id": pl.Int64, "branch_id": pl.Int64, "date_order": pl.Date,
            "sales_net": pl.Float64, "quantity": pl.Float64,
        }, orient="row",
    )


def predictions(lignes):
    return pl.DataFrame(
        lignes,
        schema={"client_id": pl.Int64, "date_score": pl.Date,
                "proba_churn": pl.Float64, "churn": pl.Int8},
        orient="row",
    )


def exemple():
    source = ventes([
        (1, 10, date(2024, 1, 1), 700, 1),
        (2, 10, date(2024, 1, 1), 1400, 1),
        (3, 20, date(2024, 1, 1), 2100, 1),
        (4, 20, date(2024, 1, 1), 700, 1),
        (1, 10, date(2024, 1, 2), 1000, -1),  # Retour exclu.
        (1, 10, date(2024, 1, 3), -500, 1),
        (1, 10, date(2024, 1, 10), 100, 1),
        (3, 20, date(2024, 1, 9), 300, 1),
        # Le début/la fin de la source ne dépendent pas des seuls clients scorés.
        (999, 99, date(2024, 1, 30), 1, 1),
    ])
    scores = predictions([
        (1, LUNDI, 0.9, 1), (2, LUNDI, 0.2, 1),
        (3, LUNDI, 0.3, 0), (4, LUNDI, 0.8, 1),
    ])
    return source, scores


def simulation(source, scores, **kwargs):
    parametres = {"fenetre_ca_j": 7, "horizon_gain_j": 7, "fraction_appels": 0.5}
    parametres.update(kwargs)
    return simuler_appels(source, scores, **parametres)


class SimulationAppelsTests(unittest.TestCase):
    def test_classement_ca_fois_risque_et_quota_par_boutique(self):
        source, scores = exemple()
        resultat = simulation(source.lazy(), scores)
        appels = resultat["appels"].sort("boutique_responsable")
        self.assertEqual(appels["client_id"].to_list(), [1, 3])
        self.assertEqual(appels["boutique_responsable"].to_list(), [10, 20])
        self.assertEqual(appels["score_priorite"].to_list(), [630, 630])
        self.assertEqual(resultat["capacites"]["nb_appels"].to_list(), [1, 1])
        self.assertNotIn("churn", appels.columns)

    def test_gain_sans_double_probabilite_et_sans_ca_deja_observe(self):
        source, scores = exemple()
        resultat = simulation(source, scores)
        evaluation = resultat["evaluation_appels"].sort("client_id")
        self.assertTrue(evaluation["evaluation_complete"].all())
        self.assertEqual(evaluation["ca_observe_horizon"].to_list(), [100, 300])
        self.assertEqual(evaluation["ca_manquant_horizon"].to_list(), [600, 1800])
        self.assertEqual(evaluation["ca_manquant_churn"].to_list(), [600, 0])
        central = resultat["scenarios"].filter(pl.col("taux_recuperation") == 0.1)
        self.assertAlmostEqual(central["ca_incremental_estime"].item(), 60)
        self.assertAlmostEqual(central["nb_recuperations_estimees"].item(), 0.1)
        nul = resultat["scenarios"].filter(pl.col("taux_recuperation") == 0)
        self.assertEqual(nul["ca_incremental_estime"].item(), 0)

    def test_les_ventes_du_lundi_et_du_futur_ne_changent_pas_la_selection(self):
        source, scores = exemple()
        avant = simulation(source, scores)
        modifiee = pl.concat([
            source,
            ventes([(1, 99, LUNDI, 100000, 1), (4, 98, LUNDI + timedelta(days=1), 1000000, 1)]),
        ])
        apres = simulation(modifiee, scores)
        assert_frame_equal(avant["appels"], apres["appels"])
        self.assertEqual(apres["evaluation_appels"].filter(pl.col("client_id") == 1)["ca_incremental_estime"].item(), 0)

    def test_ni_la_cible_ni_le_taux_ne_servent_a_choisir_les_appels(self):
        source, scores = exemple()
        avant = simulation(source, scores)
        inverse = scores.with_columns((1 - pl.col("churn")).alias("churn"))
        apres = simulation(source, inverse, taux_recuperation=0.8)
        assert_frame_equal(avant["appels"], apres["appels"])

    def test_affectation_sur_historique_complet_et_departage(self):
        source = ventes([
            (1, 10, date(2023, 12, 1), 10000, 1),
            (1, 20, date(2024, 1, 4), 100, 1),
            (2, 10, date(2024, 1, 1), 100, 1),
            (2, 20, date(2024, 1, 4), 100, 1),
            (3, 10, date(2024, 1, 1), 100, 1),
            (3, 20, date(2024, 1, 1), 100, 1),
            (999, 99, date(2024, 1, 30), 1, 1),
        ])
        scores = predictions([(i, LUNDI, 0.5, 1) for i in [1, 2, 3]])
        resultat = simulation(source, scores, fraction_appels=1)
        appels = resultat["appels"].sort("client_id")
        self.assertEqual(appels["boutique_responsable"].to_list(), [10, 20, 10])
        self.assertEqual(appels["ca_reference_client"].to_list(), [100, 200, 200])

    def test_pas_de_rappel_meme_apres_changement_de_boutique(self):
        source = ventes([
            (1, 10, date(2024, 1, 1), 100, 1),
            (1, 20, date(2024, 1, 9), 1000, 1),
            (1, 20, date(2024, 1, 16), 100, 1),
            (999, 99, date(2024, 1, 30), 1, 1),
        ])
        scores = predictions([(1, LUNDI + timedelta(days=i), 0.9, 1) for i in [0, 7, 14]])
        resultat = simulation(source, scores, fraction_appels=1, delai_rappel_j=14)
        self.assertEqual(resultat["appels"]["date_score"].to_list(), [LUNDI, LUNDI + timedelta(days=14)])
        self.assertEqual(resultat["appels"]["boutique_responsable"].to_list(), [10, 20])
        self.assertEqual(resultat["capacites"]["nb_appels"].to_list(), [1, 0, 1])

    def test_arrondi_inferieur_plafond_et_remplacement_des_clients_indisponibles(self):
        source = ventes(
            [(i, 10, date(2024, 1, 1), 100, 1) for i in range(1, 41)]
            + [(999, 99, date(2023, 11, 1), 1, 1), (999, 99, date(2024, 1, 30), 1, 1)]
        )
        scores = predictions([
            (i, lundi, i / 40, 1) for lundi in [LUNDI, LUNDI + timedelta(days=7)] for i in range(1, 41)
        ])
        resultat = simulation(
            source, scores, fraction_appels=0.05, fenetre_ca_j=30,
            delai_rappel_j=30, capacite_max_appels=1,
        )
        self.assertEqual(resultat["capacites"]["quota_clients"].to_list(), [2, 2])
        self.assertEqual(resultat["capacites"]["nb_appels"].to_list(), [1, 1])
        self.assertEqual(resultat["appels"]["client_id"].to_list(), [40, 39])
        petit = simulation(source, scores.filter(pl.col("client_id") <= 19), fraction_appels=0.05)
        self.assertTrue(petit["appels"].is_empty())
        self.assertEqual(petit["scenarios"]["ca_incremental_estime"].sum(), 0)

    def test_boutiques_non_renseignees_sont_signalees(self):
        source = ventes([(1, None, date(2024, 1, 1), 100, 1), (999, 99, date(2024, 1, 30), 1, 1)])
        resultat = simulation(source, predictions([(1, LUNDI, 0.9, 1)]), fraction_appels=1)
        self.assertTrue(resultat["appels"].is_empty())
        self.assertEqual(resultat["clients_sans_boutique"]["client_id"].to_list(), [1])

    def test_suivi_incomplet_ou_cible_inconnue_ne_donnent_pas_un_faux_gain_nul(self):
        source, scores = exemple()
        incomplet = source.filter(pl.col("date_order") <= date(2024, 1, 10))
        resultat = simulation(incomplet, scores)
        central = resultat["scenarios"].filter(pl.col("taux_recuperation") == 0.1)
        self.assertEqual(central["nb_appels_evaluables"].item(), 0)
        self.assertIsNone(central["ca_incremental_estime"].item())
        inconnus = scores.with_columns(pl.lit(None, dtype=pl.Int8).alias("churn"))
        resultat = simulation(source, inconnus)
        self.assertEqual(resultat["evaluation_appels"]["ca_manquant_churn"].null_count(), 2)

    def test_dates_chaine_et_parametres_invalides(self):
        source, scores = exemple()
        chaine = source.with_columns(pl.col("date_order").cast(pl.String))
        assert_frame_equal(simulation(source, scores)["appels"], simulation(chaine, scores)["appels"])
        for kwargs in [
            {"fraction_appels": 0}, {"taux_recuperation": 1.1},
            {"delai_rappel_j": 6}, {"capacite_max_appels": -1},
        ]:
            with self.assertRaises(ValueError):
                simulation(source, scores, **kwargs)
        with self.assertRaisesRegex(ValueError, "une seule"):
            simulation(source, pl.concat([scores, scores.head(1)]))


if __name__ == "__main__":
    unittest.main()
