import numpy as np
import pandas as pd
import pytest

from src.config.config import FEATURES, DATA_PROCESSED_DIR


def make_sample_users():
    return pd.DataFrame({
        "Num_Acc": [202100001, 202100002],
        "num_veh": [1, 1],
        "id_vehicule": [1, 2],
        "grav": [1, 3],
        "an_nais": [1985, 1970],
        "sexe": [1, 2],
        "secu1": [1.0, 2.0],
        "secu2": [0.0, 0.0],
        "secu3": [0.0, 0.0],
        "trajet": [1, 2],
        "place": [1, 2],
        "catu": [1, 1],
        "locp": [0, 0],
        "actp": [0, 0],
        "etatp": [0, 0],
    })


def make_sample_caract():
    return pd.DataFrame({
        "Num_Acc": [202100001, 202100002],
        "jour": [15, 20],
        "mois": [6, 9],
        "an": [2021, 2021],
        "hrmn": ["1430", "0900"],
        "lum": [1, 3],
        "dep": ["75", "69"],
        "com": ["75001", "69001"],
        "agg": [2, 1],
        "int": [1, 2],
        "atm": [1, 2],
        "col": [2, 3],
        "adr": ["", ""],
        "lat": ["48,8566", "45,7640"],
        "long": ["2,3522", "4,8357"],
        "catr": [3, 4],
    })


class TestConfig:
    def test_features_list_not_empty(self):
        assert len(FEATURES) == 28

    def test_features_contains_expected_keys(self):
        expected = ["place", "catu", "sexe", "vma", "lat", "long", "nb_victim", "nb_vehicules"]
        for key in expected:
            assert key in FEATURES

    def test_data_processed_dir_is_path(self):
        assert DATA_PROCESSED_DIR.name == "processed"


class TestDataTransformations:
    def test_grav_binary_mapping(self):
        # Original codes: 1=indemne, 2=blessé léger, 3=hospitalisé, 4=décédé
        # Step1: reorder -> [1,3,4,2], Step2: [2,3,4]->[0,1,1]
        # Result: 1->1, 2->3->1, 3->4->1, 4->2->0
        df = pd.DataFrame({"grav": [1, 2, 3, 4]})
        df["grav"] = df["grav"].replace([1, 2, 3, 4], [1, 3, 4, 2])
        df["grav"] = df["grav"].replace([2, 3, 4], [0, 1, 1])
        assert list(df["grav"]) == [1, 1, 1, 0]

    def test_victim_age_calculation(self):
        df = pd.DataFrame({
            "Num_Acc": [202100001, 202100002],
            "an_nais": [1990, 1980],
        })
        df["year_acc"] = df["Num_Acc"].astype(str).apply(lambda x: int(x[:4]))
        df["victim_age"] = df["year_acc"] - df["an_nais"]
        assert list(df["victim_age"]) == [31, 41]

    def test_hour_extraction_from_hrmn(self):
        # hrmn is stored as integer HHMM, e.g. 1430 = 14h30
        df = pd.DataFrame({"hrmn": [1430, 900, 2359, 0]})
        df["hour"] = df["hrmn"].astype(int) // 100
        assert list(df["hour"]) == [14, 9, 23, 0]

    def test_atm_grouping(self):
        dico = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
        df = pd.DataFrame({"atm": [1, 2, 8, 5]})
        df["atm"] = df["atm"].replace(dico)
        assert list(df["atm"]) == [0, 1, 0, 1]

    def test_lat_long_conversion(self):
        df = pd.DataFrame({"lat": ["48,8566", "45,7640"], "long": ["2,3522", "4,8357"]})
        df["lat"] = df["lat"].str.replace(",", ".").astype(float)
        df["long"] = df["long"].str.replace(",", ".").astype(float)
        assert abs(df["lat"][0] - 48.8566) < 1e-4
        assert abs(df["long"][0] - 2.3522) < 1e-4

    def test_corse_department_replacement(self):
        df = pd.DataFrame({"dep": ["2A", "2B", "75"]})
        df["dep"] = df["dep"].str.replace("2A", "201").str.replace("2B", "202")
        assert list(df["dep"]) == ["201", "202", "75"]

    def test_nan_replacement_for_minus_one(self):
        df = pd.DataFrame({"secu1": [1.0, -1.0, 2.0], "motor": [0, 1, -1]})
        df[["secu1", "motor"]] = df[["secu1", "motor"]].replace(-1, np.nan)
        assert pd.isna(df["secu1"][1])
        assert pd.isna(df["motor"][2])

    def test_train_test_split_ratio(self):
        from sklearn.model_selection import train_test_split
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        assert len(X_train) == 70
        assert len(X_test) == 30
