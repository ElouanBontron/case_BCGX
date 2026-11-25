"""Contrôle des frontières temporelles et de l'isolement du test final."""

from datetime import date, timedelta
import unittest
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

from train_churn import _choisir_seuil, _decouper_periodes, entrainer_churn


def observations():
    lignes = [
        ("train_zero", 1, date(2023, 7, 24), 0),
        ("train_un", 2, date(2023, 7, 31), 1),
        ("frontiere_train", 1, date(2023, 8, 3), 0),
        ("gap", 1, date(2023, 8, 7), 1),
        ("val_zero", 1, date(2023, 11, 6), 0),
        ("val_un", 2, date(2023, 11, 13), 1),
        ("val_decembre", 1, date(2023, 12, 25), 1),
        ("refit_janvier", 2, date(2024, 1, 1), 0),
        ("frontiere_refit", 1, date(2024, 1, 2), 1),
        ("test_zero", 1, date(2024, 4, 1), 0),
        ("test_precoce", 2, date(2024, 6, 17), 0),
        ("test_un", 2, date(2024, 6, 24), 1),
        ("apres_test", 1, date(2024, 7, 1), 1),
    ]
    return pl.DataFrame([
        {
            "cas": nom,
            "client_id": client,
            "date_score": jour,
            "fin_prediction": jour + timedelta(days=90),
            "fin_observation": date(2024, 6, 30) if nom == "test_precoce"
            else date(2024, 9, 22),
            "churn": cible,
            # Cette échéance ancienne ne doit jamais remplacer fin_prediction.
            "echeance": jour + timedelta(days=1),
            "cible_observable": True,
            "x": float(i),
            "recence_j": float(i + 1),
            "retard_relatif": float(i + 1) / 10,
            "historique_180j_complet": True,
            "evolution_ca_90j": float("inf") if i == 0 else None,
        }
        for i, (nom, client, jour, cible) in enumerate(lignes)
    ])


def decoupage(data):
    return _decouper_periodes(
        data, date(2023, 11, 1), date(2024, 1, 1),
        date(2024, 4, 1), date(2024, 7, 1),
    )


def entrainement_trace(data):
    choix, modele = Mock(), Mock()
    choix.best_iteration = 2
    choix.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    modele.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
    with patch("train_churn.XGBClassifier", side_effect=[choix, modele]) as constructeur:
        resultat = entrainer_churn(data.lazy())
    return resultat, choix, modele, constructeur


class DecoupageTemporelTests(unittest.TestCase):
    def test_frontieres_et_reutilisation_des_scores_devenus_observables(self):
        jeux = decoupage(observations())
        self.assertEqual(jeux["train"]["cas"].to_list(), ["train_zero", "train_un"])
        self.assertEqual(jeux["validation"]["cas"].to_list(), [
            "val_zero", "val_un", "val_decembre",
        ])
        self.assertEqual(jeux["train_final"]["cas"].to_list(), [
            "train_zero", "train_un", "frontiere_train", "gap",
            "val_zero", "val_un", "val_decembre", "refit_janvier",
        ])
        self.assertEqual(jeux["test"]["cas"].to_list(), ["test_zero", "test_un"])
        self.assertTrue((jeux["train"]["fin_prediction"] < date(2023, 11, 1)).all())
        self.assertTrue((jeux["validation"]["fin_prediction"] < date(2024, 4, 1)).all())
        self.assertTrue((jeux["train_final"]["fin_prediction"] < date(2024, 4, 1)).all())

    def test_un_meme_client_peut_etre_present_a_des_dates_differentes(self):
        jeux = decoupage(observations())
        self.assertEqual(set(jeux["train"]["client_id"]), {1, 2})
        self.assertEqual(set(jeux["test"]["client_id"]), {1, 2})

    def test_un_rachat_precoce_ne_rend_pas_une_fenetre_incomplete_exploitable(self):
        jeux = decoupage(observations())
        self.assertEqual(jeux["test_complet"].height, 3)
        self.assertEqual(jeux["test_complet"].filter(pl.col("cas") == "test_precoce")["churn"].item(), None)
        self.assertNotIn("test_precoce", jeux["test"]["cas"].to_list())


class EntrainementTests(unittest.TestCase):
    def test_les_modeles_ne_voient_que_leurs_periodes_et_les_variables_autorisees(self):
        resultat, choix, modele, constructeur = entrainement_trace(observations())
        self.assertEqual(resultat["variables"], [
            "x", "recence_j", "retard_relatif", "historique_180j_complet", "evolution_ca_90j",
        ])
        np.testing.assert_array_equal(choix.fit.call_args.args[0][:, 0], [0, 1])
        np.testing.assert_array_equal(
            choix.fit.call_args.kwargs["eval_set"][0][0][:, 0], [4, 5, 6]
        )
        np.testing.assert_array_equal(modele.fit.call_args.args[0][:, 0], range(8))
        np.testing.assert_array_equal(modele.predict_proba.call_args.args[0][:, 0], [9, 11])
        self.assertFalse(np.isinf(modele.fit.call_args.args[0]).any())
        self.assertTrue(np.isnan(modele.fit.call_args.args[0][:, -1]).all())
        self.assertEqual(constructeur.call_args_list[1].kwargs["n_estimators"], 3)
        self.assertNotIn("early_stopping_rounds", constructeur.call_args_list[1].kwargs)

    def test_changer_les_cibles_du_test_ne_change_pas_le_seuil_ni_lapprentissage(self):
        avant, choix_avant, modele_avant, _ = entrainement_trace(observations())
        inverse = observations().with_columns(
            pl.when(pl.col("cas").is_in(["test_zero", "test_un"]))
            .then(1 - pl.col("churn")).otherwise(pl.col("churn")).alias("churn")
        )
        apres, choix_apres, modele_apres, _ = entrainement_trace(inverse)
        self.assertEqual(avant["seuil_decision"], 0.7)
        self.assertEqual(avant["seuil_decision"], apres["seuil_decision"])
        self.assertEqual(avant["nb_arbres"], apres["nb_arbres"])
        np.testing.assert_array_equal(choix_avant.fit.call_args.args[1], choix_apres.fit.call_args.args[1])
        np.testing.assert_array_equal(modele_avant.fit.call_args.args[1], modele_apres.fit.call_args.args[1])
        np.testing.assert_array_equal(avant["matrice_confusion"], [[1, 0], [0, 1]])
        np.testing.assert_array_equal(apres["matrice_confusion"], [[0, 1], [1, 0]])

    def test_couverture_et_listes_dalertes_sont_calculees_par_date(self):
        resultat, *_ = entrainement_trace(observations())
        self.assertEqual(resultat["tailles"], {
            "train": 2, "validation": 3, "train_final": 8, "test": 2,
        })
        self.assertEqual(resultat["matrice_confusion"].sum(), 2)
        self.assertEqual(resultat["couverture_test"]["part_cibles_connues"].to_list(), [1, 0, 1])
        self.assertEqual(resultat["metriques_test"]["score"].to_list(), [
            "proba_churn", "recence_j", "retard_relatif",
        ])
        self.assertEqual(resultat["metriques_test"]["nb_alertes"].to_list(), [2, 2, 2])

    def test_dates_incoherentes_et_absence_de_fin_prediction_sont_refusees(self):
        with self.assertRaisesRegex(ValueError, "dates"):
            entrainer_churn(observations().lazy(), fin_validation=date(2023, 10, 1))
        with self.assertRaisesRegex(ValueError, "fin_prediction"):
            entrainer_churn(observations().drop("fin_prediction").lazy())

    def test_un_jeu_sans_les_deux_classes_est_refuse(self):
        data = observations().with_columns(
            pl.when(pl.col("date_score") >= date(2024, 4, 1))
            .then(0).otherwise(pl.col("churn")).alias("churn")
        )
        with self.assertRaisesRegex(ValueError, "split test"):
            entrainer_churn(data.lazy())

    def test_seuil_f1_et_departage(self):
        self.assertEqual(_choisir_seuil(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
        ), 0.35)
        self.assertEqual(_choisir_seuil(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
            np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        ), 0.9)


if __name__ == "__main__":
    unittest.main()
