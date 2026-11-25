"""Vérifie les capacités figées, l'ordre des appels et le chiffrage par épisode."""

from datetime import date, timedelta
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from churn_boutiques import ConfigurationRelances, estimer_budgets, semaines_completes
from churn_cible import construire_lot, inventorier_ventes, variables_modele
from churn_episodes import estimer_gain_ca, selectionner_relances
from churn_final import ajouter_risque_et_priorite


J = timedelta(days=1)
T = date(2024, 4, 1)


def commandes(lignes):
    return pl.LazyFrame(lignes, schema=["date_order", "branch_id", "client_id", "order_channel",
                                        "quantity", "sales_net"], orient="row")


class BudgetsTests(unittest.TestCase):
    def test_semaines_completes_actives_distincts_et_boutiques_sans_historique(self):
        debut, fin = date(2024, 1, 3), date(2024, 1, 23)
        lundi = date(2024, 1, 8)
        self.assertEqual(semaines_completes(debut, fin), (lundi, lundi + 14*J))
        self.assertEqual(semaines_completes(lundi, lundi + 6*J), (lundi, lundi + 7*J))
        lignes = [
            (lundi, 1, 1, " By Phone ", 1, 20), (lundi, 1, 1, "by phone", 1, 30),
            (lundi+J, 1, 1, "by phone", 1, 10), (lundi+6*J, 1, 2, "BY PHONE", 1, 10),
            (lundi+7*J, 1, 3, "online", 1, 10), (lundi+8*J, 1, 4, "by phone", -1, -10),
            (lundi+8*J, 1, None, "by phone", 1, 5),
            (lundi-J, 1, 5, "by phone", 1, 10), (lundi+14*J, 1, 5, "by phone", 1, 10),
            (lundi, 3, 3, "online", 1, 10), (fin+J, 4, 1, "by phone", 1, 10),
        ] + [(lundi, 2, c, "by phone", 1, 20) for c in range(40)]
        source = commandes(lignes).with_columns(pl.col("date_order").cast(pl.String))
        budgets = estimer_budgets(source, debut, fin, ConfigurationRelances())
        self.assertEqual(budgets.columns, ["branch_id", "clients_telephone_par_semaine_moy",
                                           "budget_relances_par_semaine"])
        self.assertEqual(budgets.rows(), [(1, 1.0, 0), (2, 40.0, 10), (3, 0.0, 0), (4, None, 0)])
        plus = commandes(lignes + [(fin+30*J, 2, c, "by phone", 1, 100) for c in range(80)])
        assert_frame_equal(budgets, estimer_budgets(plus, debut, fin, ConfigurationRelances()))
        double = estimer_budgets(source, debut, fin, ConfigurationRelances(coefficient_relance=0.5))
        self.assertEqual(double.filter(pl.col("branch_id") == 2)["budget_relances_par_semaine"].item(), 20)

    def test_boutique_et_ca_strictement_anterieurs_au_lundi(self):
        source = commandes([
            (T-40*J, 10, 1, "online", 1, 60), (T-20*J, 20, 1, "online", 1, 30),
            (T-20*J, 20, 1, "online", 1, 30), (T, 10, 1, "online", 1, 1000),
            (T+100*J, 10, 999, "other", 1, 1),
        ])
        lignes, _, fin = inventorier_ventes(source, ("at the store",))
        lundis, _ = construire_lot(lignes, fin)
        self.assertNotIn("branch_id", variables_modele(lundis))
        scores = lundis.filter(pl.col("date_score").is_in([T, T+7*J])).sort("date_score").collect()
        self.assertEqual(scores.select("branch_id", "ca_90j").rows(), [(20, 120.0), (10, 1120.0)])


class BacktestTests(unittest.TestCase):
    def test_ordre_local_par_produit_et_episode_global_aux_boutiques(self):
        lignes = []
        for semaine, lundi in enumerate([T, T+7*J]):
            for client, agence, ca, risque in [
                (1, 10, 10, .9), (2, 20 if semaine else 10, 100, .3),
                (3, 20, 10, .99), (4, 20, 20, .6), (5, 20, 30, .3),
                (6, 30, 100, 1.0), (7, 40, 100, 1.0), (8, None, 100, 1.0), (9, 10, 10000, 1.0),
            ]:
                achat = T if semaine and client == 4 else T-20*J
                if client == 9:
                    achat = T-90*J
                lignes.append((client, agence, achat, lundi, achat+90*J, (lundi-achat).days, 10., ca, risque))
        scores = pl.DataFrame(lignes, schema=["client_id", "branch_id", "dernier_achat", "date_score", "echeance",
                                              "recence_j", "intervalle_median_j", "ca_90j", "risque"], orient="row")
        scores = ajouter_risque_et_priorite(scores, scores["risque"])
        budgets = pl.DataFrame({"branch_id": [10, 20, 30], "budget_relances_par_semaine": [1, 2, 0]})
        appels = selectionner_relances(scores, "score_priorite", budgets)
        for lundi, attendus in [(T, [2, 3, 4]), (T+7*J, [1, 4, 5])]:
            self.assertEqual(appels.filter(pl.col("date_alerte") == lundi)["client_id"].sort().to_list(), attendus)
        self.assertEqual(appels.select(pl.struct("client_id", "dernier_achat").n_unique()).item(), appels.height)
        self.assertTrue((appels["score"] == appels["proba_churn"] * appels["ca_90j"]).all())
        regle = scores.with_columns(pl.col("recence_j").cast(pl.Float32).alias("retard_relatif"))
        comparaison = pl.concat([appels, selectionner_relances(regle, "retard_relatif", budgets)])
        self.assertEqual(comparaison.height, 11)  # La règle épuise les candidats d'une boutique en semaine 2.
        nuls = budgets.with_columns(pl.lit(0).alias("budget_relances_par_semaine"))
        self.assertTrue(selectionner_relances(scores, "score_priorite", nuls).is_empty())

    def test_aucun_plafond_global_ajoute_aux_budgets_locaux(self):
        scores = pl.DataFrame({"client_id": range(64)}).with_columns(
            branch_id=pl.col("client_id")%2, date_score=pl.lit(T), dernier_achat=pl.lit(T-J),
            echeance=pl.lit(T+89*J), recence_j=pl.lit(1), intervalle_median_j=pl.lit(10.),
            ca_90j=pl.lit(1.), proba_churn=pl.lit(.5),
        )
        budgets = pl.DataFrame({"branch_id": [0, 1], "budget_relances_par_semaine": [32, 32]})
        self.assertEqual(selectionner_relances(scores, "proba_churn", budgets).height, 64)

    def test_gain_utilise_tous_les_churns_echus_y_compris_a_un_jour(self):
        details = pl.DataFrame({
            "client_id": [1, 2, 3, 4], "date_alerte": [T]*4,
            "dernier_achat": [T-89*J, T-60*J, T-60*J, T-5*J],
            "echeance": [T+J, T+30*J, T+30*J, T+90*J],
            "issue": ["churn", "churn", "rachat", "inconnue"], "couverture_complete": [False, True, True, False],
            "ca_90j": [120000., 80000., 300000., 300000.],
        })
        gain = estimer_gain_ca(details, .10)
        self.assertEqual(gain["relances_simulees"][0], 4)
        self.assertEqual(gain["issues_inconnues"][0], 1)
        self.assertEqual(gain["episodes_churn_detectes_a_temps"][0], 2)
        self.assertEqual(gain["ca_reference_cumule_eur"][0], 200000.)
        self.assertEqual(gain["gain_ca_estime_eur"].to_list(), [10000., 20000., 40000.])
        nouveau = estimer_gain_ca(details, .12).filter("scenario_configure")
        self.assertEqual(nouveau["gain_ca_estime_eur"].item(), 24000.)
        doublon = details[1:2].with_columns(pl.lit(T+J).alias("date_alerte"), pl.lit(999999.).alias("ca_90j"))
        repete = estimer_gain_ca(pl.concat([doublon, details]), .10).filter("scenario_configure")
        self.assertEqual(repete.select("episodes_churn_detectes_a_temps", "gain_ca_estime_eur").row(0), (2, 20000.))
        self.assertEqual(estimer_gain_ca(details.clear(), .10)["gain_ca_estime_eur"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
