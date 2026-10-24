from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from neural_edsl import ImageFeatureCompiler,ProxyCompiler,SceneCompiler,compile_fields,execute,parse,render
from neural_edsl.language import fields
from neural_edsl.models import ProxyExpression,example,load_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_all_typed_programs_round_trip():
    states = list(product(range(3),range(3),range(2)))
    programs = set()
    for first,second in product(states,states):
        data = [list(first),list(second)]
        surface = render(compile_fields(data))
        assert fields(parse(surface)) == data
        programs.add(surface)
    assert len(programs) == 324


def test_bundled_examples_and_queries():
    controlled = SceneCompiler().compile(example("scene_example.npy"))
    assert render(controlled) == "(scene (object a red circle left) (object b red circle right))"
    image = ImageFeatureCompiler().compile(example("image_example.npy"))
    assert render(image) == "(scene (object a red circle left) (object b green square right))"
    assert execute(image,"select","color","red") == ("a",)
    assert execute(image,"count","shape","circle") == 1
    assert execute(image,"left_of","a","b") is True
    assert execute(image,"left_of","b","a") is False
    assert ProxyCompiler().compile(example("proxy_example.npy"),"scene").render() == "(detect scene (normal))"


@pytest.mark.parametrize("surface",[
    "(scene (object a red circle left) (object a blue square right))",
    "(scene (object a purple circle left) (object b blue square right))",
    "(scene (object a red circle left))",
    "(scene (object a red circle left) (object b blue square right)) extra",
])
def test_invalid_programs_rejected(surface):
    with pytest.raises(ValueError):
        parse(surface)


@pytest.mark.parametrize("values",[np.zeros((768,2)),np.full((2,768),np.nan),np.full((2,768),np.inf)])
def test_invalid_model_input_rejected(values):
    with pytest.raises(ValueError):
        ImageFeatureCompiler().compile(values)


def test_proxy_type_constraints():
    with pytest.raises(ValueError):
        ProxyExpression("command","scene","normal")
    with pytest.raises(ValueError):
        ProxyCompiler().compile(example("proxy_example.npy"),"unknown")


def test_image_benchmark_matches_shipped_evidence():
    data = np.load(ROOT / "benchmarks/image_features.npz",allow_pickle=False)["features"]
    states = list(product(range(3),range(3),range(2)))
    lookup = {(tuple(state),style):style*18+i for style in range(8) for i,state in enumerate(states)}
    meanings = [np.array([[ca,sa,p],[cb,sb,1-p]]) for ca,sa,cb,sb,p in product(range(3),range(3),range(3),range(3),range(2))
                if (ca,sa) == (0,0) or (cb,sb) == (0,0)]
    evidence = json.loads((ROOT / "results/evidence.json").read_text())["image"]
    for kind in ("logistic","semantic_grid"):
        parameters = load_parameters("image.json")
        parameters["selected"] = kind
        compiler = ImageFeatureCompiler(parameters)
        correct = 0
        for meaning in meanings:
            for first,second in product((6,7),repeat=2):
                values = data[[lookup[(tuple(meaning[0]),first)],lookup[(tuple(meaning[1]),second)]]]
                correct += fields(compiler.compile(values)) == meaning.tolist()
        assert correct == evidence["models"][kind]["heldout_correct"]
