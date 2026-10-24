"""Portable NumPy inference for the fitted compiler front ends.

Each model expects the representation it was trained against. The typed layer
validates predicted fields; correctness relative to the input is evaluated
separately in the accompanying experiments.
"""
from dataclasses import dataclass
from importlib import resources
import json
import re

import numpy as np

from .language import compile_fields


def asset(name):
    return resources.files("neural_edsl").joinpath("assets",name)


def load_parameters(name):
    return json.loads(asset(name).read_text(encoding="utf-8"))


def example(name):
    with asset(name).open("rb") as stream:
        return np.load(stream,allow_pickle=False)


def finite_array(values,shape):
    values = np.asarray(values,dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite {shape} array")
    return values


@dataclass(frozen=True)
class ProxyExpression:
    operation: str
    domain: str
    fact: str

    def __post_init__(self):
        allowed = {("detect","scene"):{"normal","anomaly"},
                   ("command","robot"):{"proceed","abort"}}
        if self.fact not in allowed.get((self.operation,self.domain),set()):
            raise ValueError("Invalid proxy expression")

    @classmethod
    def parse(cls,text):
        match = re.fullmatch(r"\(\s*(\w+)\s+(\w+)\s+\(\s*(\w+)\s*\)\s*\)",text)
        if match is None:
            raise ValueError("Malformed proxy expression")
        return cls(*match.groups())

    def render(self):
        return f"({self.operation} {self.domain} ({self.fact}))"


class SceneCompiler:
    """Three learned scalar coordinates -> discrete color, shape, position."""
    def __init__(self,parameters=None):
        self.parameters = parameters if parameters is not None else load_parameters("scene.json")

    def fields(self,values):
        p = self.parameters
        values = finite_array(values,(2,p["input_dim"]))
        coordinates = values@np.asarray(p["weights"]).T+np.asarray(p["bias"])
        return np.stack([np.abs(coordinates[:,i,None]-np.asarray(levels)).argmin(axis=1)
                         for i,levels in enumerate(p["levels"])],axis=1).tolist()

    def compile(self,values):
        return compile_fields(self.fields(values))


class ImageFeatureCompiler:
    """Frozen DINOv2 object features -> the same typed scene language."""
    def __init__(self,parameters=None):
        self.parameters = parameters if parameters is not None else load_parameters("image.json")

    def fields(self,values):
        p = self.parameters
        values = finite_array(values,(2,p["input_dim"]))
        standardized = (values-np.asarray(p["mean"]))/np.asarray(p["scale"])
        if p["selected"] == "logistic":
            result = []
            for head in p["logistic"]:
                scores = standardized@np.asarray(head["coefficients"]).T+np.asarray(head["intercept"])
                # scikit-learn stores a single signed score for two classes.
                chosen = (scores[:,0] > 0).astype(int) if scores.shape[1] == 1 else scores.argmax(axis=1)
                result.append(np.asarray(head["classes"])[chosen])
            return np.stack(result,axis=1).astype(int).tolist()
        head = p["semantic_grid"]
        coordinates = standardized@np.asarray(head["coefficients"]).T+np.asarray(head["intercept"])
        return np.stack([np.abs(coordinates[:,i,None]-np.asarray(levels)).argmin(axis=1)
                         for i,levels in enumerate(head["levels"])],axis=1).tolist()

    def compile(self,values):
        return compile_fields(self.fields(values))


class ProxyCompiler:
    """Four-class readout for recovered video/robot proxy features.

    The supplied domain is a feature hint, not a restriction on output classes.
    A returned expression satisfies the proxy schema; its fidelity to the input
    is a separate empirical question.
    """
    def __init__(self,parameters=None):
        self.parameters = parameters if parameters is not None else load_parameters("proxy.json")

    def compile(self,values,domain):
        if domain not in ("scene","robot"):
            raise ValueError("Domain must be scene or robot")
        # Preserve the float32 per-array normalization used during fitting.
        values = finite_array(values,(17,16))
        with np.errstate(over="ignore",invalid="ignore"):
            values = values.astype(np.float32)
            mean,scale = values.mean(),values.std()
            normalized = (values-mean)/(scale+1e-8)
        if (not np.isfinite(values).all() or not np.isfinite(mean)
                or not np.isfinite(scale) or not np.isfinite(normalized).all()):
            raise ValueError("Proxy input exceeds the supported float32 normalization range")
        features = np.concatenate([normalized.reshape(-1),[1.,0.] if domain == "scene" else [0.,1.]])
        p = self.parameters
        standardized = (features-np.asarray(p["mean"]))/np.asarray(p["scale"])
        scores = standardized@np.asarray(p["coefficients"]).T+np.asarray(p["intercept"])
        return ProxyExpression.parse(p["expressions"][int(scores.argmax())])
