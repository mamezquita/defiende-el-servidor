import pandas as pd
import pytest

from dsrv.config import load_yaml
from dsrv.select import (
    RULE_CONSTRAINT,
    RULE_DEFAULT,
    RULE_FALLBACK,
    apply_policy,
    choose_best,
    compare_with_v1,
)


@pytest.fixture
def policy():
    return load_yaml("configs/threshold_policy.yaml")


def sweep(rows):
    return pd.DataFrame(rows, columns=["threshold", "recall_attack", "fpr", "macro_f1"])


def test_real_policy_has_the_agreed_values(policy):
    assert policy["objetivo"]["maximizar"] == "recall_attack"
    assert policy["restriccion"]["metrica"] == "fpr"
    assert policy["restriccion"]["maximo"] == 0.05
    assert policy["desempate"] == "macro_f1"
    assert policy["respaldo"]["recall_attack_minimo"] == 0.90
    assert policy["respaldo"]["umbral_por_defecto"] == 0.5


def test_maximizes_recall_within_the_fpr_limit(policy):
    data = sweep([[0.3, 0.99, 0.20, 0.8], [0.5, 0.95, 0.05, 0.9], [0.7, 0.80, 0.01, 0.85]])
    row, rule = apply_policy(data, policy)

    assert row["threshold"] == 0.5
    assert rule == RULE_CONSTRAINT


def test_fpr_limit_is_inclusive(policy):
    row, rule = apply_policy(sweep([[0.5, 0.9, 0.05, 0.9], [0.6, 0.8, 0.04, 0.9]]), policy)

    assert row["threshold"] == 0.5 and rule == RULE_CONSTRAINT


def test_breaks_recall_ties_with_macro_f1(policy):
    row, _ = apply_policy(sweep([[0.5, 0.9, 0.04, 0.80], [0.6, 0.9, 0.03, 0.85]]), policy)

    assert row["threshold"] == 0.6


def test_last_tie_goes_to_the_lowest_threshold(policy):
    row, _ = apply_policy(sweep([[0.6, 0.9, 0.04, 0.8], [0.5, 0.9, 0.04, 0.8]]), policy)

    assert row["threshold"] == 0.5


def test_fallback_takes_lowest_fpr_with_enough_recall(policy):
    data = sweep([[0.3, 0.99, 0.30, 0.7], [0.5, 0.91, 0.10, 0.9], [0.7, 0.80, 0.08, 0.9]])
    row, rule = apply_policy(data, policy)

    assert row["threshold"] == 0.5
    assert rule == RULE_FALLBACK


def test_default_threshold_when_nothing_qualifies(policy):
    data = sweep([[0.3, 0.80, 0.30, 0.7], [0.5, 0.70, 0.20, 0.8], [0.7, 0.60, 0.10, 0.9]])
    row, rule = apply_policy(data, policy)

    assert row["threshold"] == 0.5
    assert rule == RULE_DEFAULT


def test_choose_best_prefers_a_config_that_meets_the_constraint(policy):
    table = pd.DataFrame(
        {
            "config_id": ["a", "b", "c"],
            "recall_attack": [0.99, 0.85, 0.90],
            "fpr": [0.10, 0.04, 0.05],
            "macro_f1": [0.9, 0.9, 0.8],
            "regla": [RULE_FALLBACK, RULE_CONSTRAINT, RULE_CONSTRAINT],
        }
    )

    assert choose_best(table, policy)["config_id"] == "c"


def test_choose_best_uses_lowest_fpr_among_fallbacks(policy):
    table = pd.DataFrame(
        {
            "config_id": ["a", "b"],
            "recall_attack": [0.99, 0.91],
            "fpr": [0.10, 0.07],
            "macro_f1": [0.9, 0.8],
            "regla": [RULE_FALLBACK, RULE_FALLBACK],
        }
    )

    assert choose_best(table, policy)["config_id"] == "b"


def test_compare_with_v1_treats_lower_fpr_as_improvement():
    base = {"precision_attack": 0.8, "macro_f1": 0.8, "accuracy": 0.8, "tn": 90, "tp": 90}
    v1 = pd.Series({**base, "recall_attack": 0.98, "fpr": 0.18, "fp": 18, "fn": 2})
    v2 = pd.Series({**base, "recall_attack": 0.90, "fpr": 0.05, "fp": 5, "fn": 10})
    comparison = compare_with_v1(v1, v2)

    assert any(text.startswith("fpr") for text in comparison["mejora"])
    assert any(text.startswith("recall_attack") for text in comparison["empeora"])
    assert any(text.startswith("falsos positivos") for text in comparison["mejora"])
    assert any(text.startswith("falsos negativos") for text in comparison["empeora"])
    assert any(text.startswith("accuracy") for text in comparison["igual"])
