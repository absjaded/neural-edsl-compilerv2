"""Original-input path: numerical rejection and unchanged fitted predictions."""
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from neural_edsl import ProxyCompiler
from neural_edsl.cli import main
from neural_edsl.models import example, load_parameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("values", [
    np.full((17, 16), 1e300),  # Finite float64, but cannot be cast to float32.
    np.full((17, 16), 3e38),   # Finite float32 entries; mean reduction overflows.
    np.tile([1e20, -1e20], (17, 8)),  # Finite mean; variance reduction overflows.
])
def test_proxy_rejects_normalization_overflow(values):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="float32 normalization"):
            ProxyCompiler().compile(values, "scene")


@pytest.mark.parametrize("values", [
    np.zeros((16, 17)),
    np.full((17, 16), np.nan),
    np.full((17, 16), np.inf),
])
def test_proxy_rejects_malformed_input(values):
    with pytest.raises(ValueError, match="finite"):
        ProxyCompiler().compile(values, "scene")


@pytest.mark.parametrize("value", [0., 1.])
def test_constant_proxy_input_remains_defined(value):
    # Constant inputs remain an existing control, not a new abstention policy.
    result = ProxyCompiler().compile(np.full((17, 16), value), "scene")
    assert result.render() in load_parameters("proxy.json")["expressions"]


def test_proxy_domain_is_a_feature_hint():
    # Preserve the fitted model: the hint does not mask its four output classes.
    result = ProxyCompiler().compile(example("proxy_example.npy"), "robot")
    assert result.render() == "(detect scene (normal))"


def test_proxy_preserves_legacy_predictions_on_all_bundled_rows():
    parameters = load_parameters("proxy.json")
    compiler = ProxyCompiler(parameters)
    with np.load(ROOT / "benchmarks/proxy_features.npz", allow_pickle=False) as data:
        raw, domains = data["features"], data["domains"]
    assert len(raw) == 340
    for values, domain in zip(raw, domains):
        # Reference arithmetic from the original released NumPy front end.
        original = np.asarray(values, dtype=np.float64).astype(np.float32)
        normalized = (original - original.mean()) / (original.std() + 1e-8)
        hint = [1., 0.] if domain == 0 else [0., 1.]
        features = np.concatenate([normalized.reshape(-1), hint])
        standardized = (features - np.asarray(parameters["mean"])) / np.asarray(parameters["scale"])
        scores = standardized @ np.asarray(parameters["coefficients"]).T + np.asarray(parameters["intercept"])
        expected = parameters["expressions"][int(scores.argmax())]
        assert compiler.compile(values, ("scene", "robot")[int(domain)]).render() == expected


@pytest.mark.parametrize("arguments, expected", [
    (["proxy", "--domain", "scene"], "(detect scene (normal))"),
    (["scene", "--query", "count", "shape", "circle"], 2),
    (["image-features", "--query", "select", "color", "red"], ["a"]),
])
def test_documented_cli_paths(arguments, expected, capsys):
    main(arguments)
    result = json.loads(capsys.readouterr().out)
    key = "program" if arguments[0] == "proxy" else "answer"
    assert result[key] == expected


def test_cli_reports_proxy_normalization_error(tmp_path, capsys):
    path = tmp_path / "oversized.npy"
    np.save(path, np.full((17, 16), 1e300), allow_pickle=False)
    with pytest.raises(SystemExit) as caught:
        main(["proxy", "--input", str(path)])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "float32 normalization" in captured.err
    assert not captured.out
