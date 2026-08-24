from explainability.shap_explainer import explain_predictions


def test_shap_explanation():
    importance = explain_predictions(
        n=100,
        seed=42,
    )

    assert len(importance) == 7
    assert all(value >= 0 for value in importance.values())
