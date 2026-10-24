"""Controlled continuous-to-symbolic composition experiment (see PLAN.md)."""
from __future__ import annotations

import copy
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neural_edsl.language import compile_fields,fields,parse,query_signature,render

ROOT = Path(__file__).resolve().parent


def meanings():
    return np.array([[[ca,sa,p],[cb,sb,1-p]] for ca,sa,cb,sb,p in product(range(3),range(3),range(3),range(3),range(2))],dtype=np.int64)


def heldout_mask(labels):
    return ((labels[:,:,0] == 0)&(labels[:,:,1] == 0)).any(axis=1)


def mixing_matrix():
    random = np.random.default_rng(20261022)
    matrix,_ = np.linalg.qr(random.standard_normal((32,13)))
    return matrix.astype(np.float32)


def encode(labels,rng,noisy=True):
    raw = np.zeros((len(labels),2,13),dtype=np.float32)
    for i in range(len(labels)):
        for role in range(2):
            color,shape,position = labels[i,role]
            raw[i,role,color] = 1
            raw[i,role,3+shape] = 1
            raw[i,role,6] = -0.75 if position == 0 else 0.75
    if noisy:
        raw[:,:,6] += rng.uniform(-0.05,0.05,size=raw[:,:,6].shape)
        raw[:,:,7:] = rng.normal(0,0.25,size=raw[:,:,7:].shape)
    encoded = raw@mixing_matrix().T
    if noisy:
        encoded += rng.normal(0,0.01,size=encoded.shape)
    return encoded.astype(np.float32)


def dataset(labels,views,seed):
    repeated = np.repeat(labels,views,axis=0)
    return torch.tensor(encode(repeated,np.random.default_rng(seed))),torch.tensor(repeated)


class Compiler(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.readout = nn.Linear(32,8)
        elif kind in ("quantized","continuous_bottleneck"):
            self.encoder = nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,4))
            self.readout = nn.Linear(4,8)
        elif kind == "factor_grid":
            self.encoder = nn.Linear(32,3)
        else:
            raise ValueError("Unknown compiler type")

    def codes(self,values):
        # Five scalar values on each of four axes: at most 5**4 codes.
        bounded = torch.tanh(self.encoder(values))
        if self.kind == "continuous_bottleneck":
            return bounded
        hard = (bounded*2).round().clamp(-2,2)/2
        return bounded+(hard-bounded).detach() if self.training else hard

    def forward(self,values):
        if self.kind == "factor_grid":
            coordinates = self.encoder(values)
            return tuple(-10*(coordinates[...,i,None]-coordinates.new_tensor(centers))**2
                         for i,centers in enumerate(([-1.,0.,1.],[-1.,0.,1.],[-1.,1.])))
        hidden = self.codes(values) if self.kind in ("quantized","continuous_bottleneck") else values
        logits = self.readout(hidden)
        return logits[...,:3],logits[...,3:6],logits[...,6:]


def predict(model,values):
    with torch.inference_mode():
        return torch.stack([scores.argmax(-1) for scores in model(values)],dim=-1).cpu().numpy()


def loss(model,values,expected):
    return sum(F.cross_entropy(scores.reshape(-1,scores.shape[-1]),expected[...,i].reshape(-1))
               for i,scores in enumerate(model(values)))/3


def metrics(predicted,expected):
    predicted = np.asarray(predicted)
    expected = np.asarray(expected)
    correct = (predicted == expected)
    all_queries,query_correct = 0,0
    all_program_queries = 0
    programs = []
    for got,wanted in zip(predicted,expected):
        program = compile_fields(got.tolist())
        source = compile_fields(wanted.tolist())
        surface = render(program)
        assert fields(parse(surface)) == fields(program)
        actual_queries,target_queries = query_signature(program),query_signature(source)
        same = [a == b for a,b in zip(actual_queries,target_queries)]
        query_correct += sum(same)
        all_queries += len(same)
        all_program_queries += all(same)
        programs.append(dict(prediction=surface,target=render(source)))
    return dict(n=len(expected),exact_correct=int(correct.all(axis=(1,2)).sum()),
                exact_accuracy=float(correct.all(axis=(1,2)).mean()),
                field_accuracy=float(correct.mean()),
                query_accuracy=query_correct/all_queries,
                all_queries_correct=all_program_queries,
                grammar_and_type_acceptance=1.0,programs=programs)


def collision_certificate(labels):
    features = encode(labels,np.random.default_rng(0),noisy=False)
    lookup = {tuple(label.reshape(-1)):i for i,label in enumerate(labels)}
    seen = set()
    pairs = []
    for i,label in enumerate(labels):
        if i in seen:
            continue
        other = lookup[tuple(label[::-1].reshape(-1))]
        assert other != i
        assert np.array_equal(features[i].mean(axis=0),features[other].mean(axis=0))
        assert render(compile_fields(label.tolist())) != render(compile_fields(labels[other].tolist()))
        pairs.append([i,other])
        seen.update((i,other))
    assert len(seen) == len(labels)
    return dict(programs=len(labels),pairs=len(pairs),exact_pooled_collisions=len(pairs),
                deterministic_exact_match_ceiling=0.5,pair_indices=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds",nargs="+",choices=["linear","quantized","continuous_bottleneck","factor_grid"],default=["linear","quantized"])
    parser.add_argument("--output",default="run")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = ROOT / args.output
    if output.resolve().parent != ROOT:
        raise ValueError("Output must be a child directory")
    output.mkdir(exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("Preserve completed experiment results")
    labels = meanings()
    heldout = heldout_mask(labels)
    train_labels,test_labels = labels[~heldout],labels[heldout]
    assert len(train_labels) == 128 and len(test_labels) == 34
    assert not any(tuple(obj[:2]) == (0,0) for scene in train_labels for obj in scene)
    x_train,y_train = dataset(train_labels,12,100)
    x_val,y_val = dataset(train_labels,3,200)
    x_test,y_test = dataset(test_labels,16,300)
    x_clean = torch.tensor(encode(test_labels,np.random.default_rng(0),noisy=False))
    np.save(output / "mixing_matrix.npy",mixing_matrix())
    np.save(output / "example_input.npy",x_clean[0].numpy())
    (output / "split.json").write_text(json.dumps(dict(training_meanings=train_labels.tolist(),heldout_meanings=test_labels.tolist(),
                                                      excluded_combination="red circle on either role",train_views=12,validation_views=3,test_views=16),indent=2))
    results = []
    started = time.perf_counter()
    for seed in (11,29,47):
        for kind in args.kinds:
            torch.manual_seed(seed)
            model = Compiler(kind)
            optimizer = torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.0001)
            best,best_epoch,best_loss = copy.deepcopy(model.state_dict()),0,float("inf")
            history = []
            for epoch in range(1,151):
                model.train()
                order = torch.randperm(len(x_train))
                for low in range(0,len(order),128):
                    selected = order[low:low+128]
                    optimizer.zero_grad(set_to_none=True)
                    objective = loss(model,x_train[selected],y_train[selected])
                    objective.backward()
                    optimizer.step()
                if epoch % 5 == 0:
                    model.eval()
                    with torch.inference_mode():
                        validation = float(loss(model,x_val,y_val))
                    history.append(dict(epoch=epoch,validation_loss=validation))
                    if validation < best_loss-1e-5:
                        best,best_epoch,best_loss = copy.deepcopy(model.state_dict()),epoch,validation
                    if epoch-best_epoch >= 25:
                        break
            model.load_state_dict(best)
            model.eval()
            checkpoint = output / f"{kind}_{seed}.pt"
            torch.save(dict(kind=kind,state=best),checkpoint)
            clean_prediction = predict(model,x_clean)
            noisy_prediction = predict(model,x_test)
            shuffled_order = np.random.default_rng(401).permutation(len(x_clean))
            swapped_prediction = predict(model,x_clean.flip(1))
            counterfactual_x = x_clean.clone()
            counterfactual_labels = test_labels.copy()
            for i in range(len(test_labels)):
                old = test_labels[i,0,0]
                new = (old+1)%3
                counterfactual_labels[i,0,0] = new
                delta = np.zeros(13,dtype=np.float32)
                delta[old] -= 1
                delta[new] += 1
                counterfactual_x[i,0] += torch.tensor(delta@mixing_matrix().T)
            counterfactual_prediction = predict(model,counterfactual_x)
            unchanged = np.ones((2,3),dtype=bool)
            unchanged[0,0] = False
            report = dict(kind=kind,seed=seed,best_epoch=best_epoch,best_validation_loss=best_loss,
                          clean=metrics(clean_prediction,test_labels),noisy=metrics(noisy_prediction,y_test.numpy()),
                          mismatched=metrics(predict(model,x_clean[shuffled_order]),test_labels),
                          role_swap=metrics(swapped_prediction,test_labels[:,::-1]),
                          role_swap_equivariance=float((swapped_prediction == clean_prediction[:,::-1]).all(axis=(1,2)).mean()),
                          counterfactual=metrics(counterfactual_prediction,counterfactual_labels),
                          counterfactual_other_fields_unchanged=float((counterfactual_prediction[:,unchanged] == clean_prediction[:,unchanged]).all(axis=1).mean()),
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),history=history)
            if kind == "quantized":
                with torch.inference_mode():
                    codes = model.codes(x_clean).reshape(-1,4)
                    assert torch.all((codes*2).round() == codes*2)
                    report["observed_codes"] = len(torch.unique(codes,dim=0))
                    report["maximum_codes"] = 625
            if kind == "factor_grid":
                report["maximum_codes_per_object"] = 18
                report["observed_codes"] = len(np.unique(clean_prediction.reshape(-1,3),axis=0))
            results.append(report)
            print(json.dumps(dict(kind=kind,seed=seed,best_epoch=best_epoch,
                                  clean_accuracy=report["clean"]["exact_accuracy"],noisy_accuracy=report["noisy"]["exact_accuracy"],
                                  counterfactual_accuracy=report["counterfactual"]["exact_accuracy"])),flush=True)
    report = dict(train_meanings=len(train_labels),heldout_meanings=len(test_labels),
                  full_collision_certificate=collision_certificate(labels),
                  heldout_collision_certificate=collision_certificate(test_labels),
                  results=results,seconds=time.perf_counter()-started)
    (output / "results.json").write_text(json.dumps(report,indent=2))
    print(f"Saved composition results ({report['seconds']:.1f}s)",flush=True)


if __name__ == "__main__":
    main()
