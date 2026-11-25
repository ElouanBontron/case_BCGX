"""Vérifications des dates, de la censure et des agrégats utilisés pour le churn."""

from datetime import date, timedelta
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from churn_features import construire_lf_churn, construire_quotidien


SCORE = date(2024, 7, 1)  # Un lundi.
RECOIL = SCORE + timedelta(days=120)


def historique(client=1):
    jours = [-200, -180, -170, -140, -100, -90, -70, -40, -20, -1]
    return [(client, SCORE + timedelta(days=j), float(10 * (i + 1)))
            for i, j in enumerate(jours)]


def quotidien(lignes):
    return pl.DataFrame(
        lignes, schema=[("client_id", pl.Int64), ("date", pl.Date), ("ca_achats", pl.Float64)],
        orient="row",
    ).lazy()


def scores(lignes, fin=RECOIL, **kwargs):
    return construire_lf_churn(
        quotidien(lignes), fin_observation=fin, **kwargs
    ).filter(pl.col("date_score") == SCORE).collect()


class ConstructionChurnTests(unittest.TestCase):
    def test_fenetres_et_regularite_correspondent_aux_achats_passes(self):
        lignes = historique()
        # Le score du matin ne doit pas utiliser l'achat de ce lundi.
        lignes.append((1, SCORE, 99999.0))
        ligne = scores(lignes).row(0, named=True)
        self.assertEqual(ligne["nb_jours_achat"], 10)
        self.assertEqual(ligne["anciennete_j"], 200)
        self.assertEqual(ligne["recence_j"], 1)
        for jours, decalage, suffixe in (
            (30, 0, "30j"), (90, 0, "90j"), (90, 90, "90j_precedents"),
            (180, 0, "180j"), (365, 0, "365j"),
        ):
            retenues = [r for r in lignes if
                        SCORE - timedelta(days=jours + decalage) <= r[1]
                        < SCORE - timedelta(days=decalage)]
            self.assertEqual(ligne[f"nb_jours_achat_{suffixe}"], len(retenues))
            self.assertEqual(ligne[f"ca_{suffixe}"], sum(r[2] for r in retenues))
        self.assertAlmostEqual(ligne["evolution_frequence_90j"], 5 / 4 - 1)
        self.assertEqual(ligne["churn"], 0)
        self.assertGreater(ligne["cv_intervalles"], 0)

    def test_modifier_le_futur_ne_change_aucune_variable(self):
        sans_retour = scores(historique())
        avec_retour = scores(historique() + [(1, SCORE, 999999.0), (1, SCORE + timedelta(days=1), 1.0)])
        self.assertEqual(sans_retour["churn"].item(), 1)
        self.assertEqual(avec_retour["churn"].item(), 0)
        assert_frame_equal(sans_retour.drop("churn"), avec_retour.drop("churn"))

    def test_les_deux_classes_attendent_la_fin_du_meme_horizon(self):
        lignes = historique(1) + historique(2) + [(1, SCORE, 5.0)]
        incomplet = scores(lignes, fin=SCORE + timedelta(days=89))
        self.assertEqual(incomplet.height, 2)
        self.assertEqual(incomplet["churn"].null_count(), 2)
        self.assertFalse(incomplet["cible_observable"].any())
        complet = scores(lignes, fin=SCORE + timedelta(days=90)).sort("client_id")
        self.assertEqual(complet["churn"].to_list(), [0, 1])

    def test_rachat_le_jour_de_lecheance_est_inclus(self):
        echeance = SCORE + timedelta(days=89)
        a_temps = scores(historique() + [(1, echeance, 10.0)])
        trop_tard = scores(historique() + [(1, echeance + timedelta(days=1), 10.0)])
        self.assertEqual(a_temps["echeance"].item(), echeance)
        self.assertEqual(a_temps["churn"].item(), 0)
        self.assertEqual(trop_tard["churn"].item(), 1)

    def test_les_clients_quotidiens_ont_des_scores(self):
        debut = date(2024, 1, 1)
        lignes = [(1, debut + timedelta(days=i), 100.0)
                  for i in range(90) if (debut + timedelta(days=i)).weekday() < 5]
        resultat = construire_lf_churn(
            quotidien(lignes), fin_observation=date(2024, 6, 30)
        ).collect()
        self.assertGreater(resultat.height, 0)
        self.assertEqual(resultat["delai_churn_j"].unique().to_list(), [90])
        self.assertTrue((resultat["dernier_achat"] < resultat["date_score"]).all())

    def test_rythme_lent_et_nombre_minimum_dachats(self):
        lignes = [(1, SCORE + timedelta(days=-590 + i * 80), 10.0) for i in range(6)]
        lignes += [(2, SCORE + timedelta(days=-500 + i * 80), 10.0) for i in range(6)]
        lignes += historique(3)[:5] + [(3, SCORE + timedelta(days=1), 10.0)]
        resultat = scores(lignes)
        self.assertEqual(resultat["client_id"].to_list(), [1])
        self.assertEqual(resultat["delai_churn_j"].item(), 240)
        self.assertEqual(resultat["delai_avant_echeance_j"].item(), 50)

    def test_tendances_sans_historique_ou_denominateur_restent_nulles(self):
        recent = [(1, SCORE + timedelta(days=i), 10.0) for i in [-6, -5, -4, -3, -2, -1]]
        sans_base = [(2, SCORE + timedelta(days=i), 10.0) for i in [-200, -5, -4, -3, -2, -1]]
        resultat = scores(recent + sans_base)
        self.assertEqual(resultat["evolution_ca_90j"].null_count(), 2)
        self.assertEqual(resultat["evolution_frequence_90j"].null_count(), 2)
        for nom, dtype in resultat.schema.items():
            if dtype.is_float():
                self.assertFalse(resultat[nom].is_infinite().any(), nom)
                self.assertFalse(resultat[nom].is_nan().any(), nom)

    def test_preparation_exclut_retours_et_montants_invalides(self):
        ventes = pl.LazyFrame({
            "client_id": [1, 1, 1, 1, 1, 1, None],
            "date_order": ["2024-01-01"] * 7,
            "quantity": [1, 1, -1, 1, 1, 0, 1],
            "sales_net": [100.0, 50.0, 70.0, -20.0, float("nan"), 80.0, 10.0],
            "product_id": [10, 20, 30, 40, 50, 60, 70],
            "branch_id": [2] * 7,
            "order_channel": [" ONLINE ", "by phone", "online", "other", "other", "other", "other"],
        })
        ligne = construire_quotidien(ventes).collect().row(0, named=True)
        self.assertEqual(ligne["date"], date(2024, 1, 1))
        self.assertEqual(ligne["ca_achats"], 150.0)
        self.assertEqual(ligne["nb_lignes"], 2)
        self.assertEqual(set(ligne["produits"]), {10, 20})
        self.assertEqual(set(ligne["canaux"]), {"online", "by phone"})
        self.assertEqual(ligne["ca_online"], 100.0)

    def test_diversite_et_canaux_facultatifs_excluent_le_futur(self):
        lignes = historique() + [(1, SCORE, 10000.0)]
        ventes = pl.LazyFrame({
            "client_id": [r[0] for r in lignes],
            "date_order": [r[1] for r in lignes],
            "sales_net": [r[2] for r in lignes],
            "quantity": [1] * len(lignes),
            "product_id": [i % 3 for i in range(10)] + [999],
            "branch_id": [1] * 10 + [999],
            "order_channel": ["online"] * 10 + ["other"],
        })
        resultat = construire_lf_churn(
            construire_quotidien(ventes), fin_observation=RECOIL
        ).filter(pl.col("date_score") == SCORE).collect()
        ligne = resultat.row(0, named=True)
        self.assertEqual(ligne["nb_produits_90j"], 3)
        self.assertEqual(ligne["nb_agences_90j"], 1)
        self.assertEqual(ligne["nb_canaux_90j"], 1)
        self.assertEqual(ligne["part_ca_online_90j"], 1.0)
        self.assertNotIn("nb_produits_90j", scores(historique()).columns)

    def test_cles_dupliquees_ordre_et_source_vide(self):
        lignes = historique()
        reference = scores(lignes)
        lignes[-1] = (1, lignes[-1][1], lignes[-1][2] / 2)
        lignes.append(lignes[-1])
        assert_frame_equal(reference, scores(list(reversed(lignes))))
        self.assertTrue(construire_lf_churn(quotidien([])).collect().is_empty())


if __name__ == "__main__":
    unittest.main()
