"""Frozen DINOv2 features -> learned fields -> executable programs; see PLAN.md."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image,ImageDraw

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT.parent
CACHE = Path(os.environ.get("TORCH_HOME", str(Path.home()/".cache/torch")))/"hub"
REPOSITORY = Path(os.environ.get("DINO_REPOSITORY", str(CACHE/"facebookresearch_dinov2_main")))
WEIGHTS = Path(os.environ.get("DINO_WEIGHTS", str(CACHE/"checkpoints/dinov2_vitb14_reg4_pretrain.pth")))
COLORS = ((205,48,45),(48,165,75),(45,85,205))
STATES = np.array(list(product(range(3),range(3),range(2))),dtype=np.int64)


def render_card(state,style):
    rng = np.random.default_rng(20261100+style)
    background = tuple(int(v) for v in rng.integers(220,246,size=3))
    image = Image.new("RGB",(224,224),background)
    draw = ImageDraw.Draw(image)
    color,shape,position = map(int,state)
    center_x = (68 if position == 0 else 156)+int(rng.integers(-6,7))
    center_y = 112+int(rng.integers(-9,10))
    radius = int(rng.integers(33,43))
    box = (center_x-radius,center_y-radius,center_x+radius,center_y+radius)
    fill = COLORS[color]
    if shape == 0:
        draw.ellipse(box,fill=fill)
    elif shape == 1:
        draw.rectangle(box,fill=fill)
    else:
        draw.polygon([(center_x,center_y-radius),(center_x-radius,center_y+radius),
                      (center_x+radius,center_y+radius)],fill=fill)
    return image


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def extract(output):
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(REPOSITORY))
    from dinov2.hub.backbones import dinov2_vitb14_reg
    saved = output / "features.npz"
    records = [dict(state=state.tolist(),style=style) for style in range(8) for state in STATES]
    if saved.exists():
        data = np.load(saved,allow_pickle=False)
        if data["features"].shape != (144,768):
            raise ValueError("Unexpected cached feature shape")
        return data["features"],records
    model = dinov2_vitb14_reg(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    print("DINOv2 weights loaded strictly; extracting 144 object cards on CPU",flush=True)
    all_features = []
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std = torch.tensor([0.229,0.224,0.225])[:,None,None]
    images = output / "images"
    images.mkdir(exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        for low in range(0,len(records),8):
            tensors = []
            for index,record in enumerate(records[low:low+8],start=low):
                image = render_card(record["state"],record["style"])
                image.save(images / f"card_{index:03d}.png")
                array = np.asarray(image,dtype=np.float32)/255
                tensors.append((torch.from_numpy(array).permute(2,0,1)-mean)/std)
            features = model(torch.stack(tensors)).cpu().numpy()
            all_features.append(features)
            print(f"Features {min(low+8,len(records))}/144 ({time.perf_counter()-started:.1f}s)",flush=True)
    features = np.concatenate(all_features)
    assert features.shape == (144,768) and np.isfinite(features).all()
    np.savez_compressed(saved,features=features)
    revision = subprocess.run(["git","-C",str(REPOSITORY),"rev-parse","HEAD"],capture_output=True,text=True)
    provenance = dict(weights=str(WEIGHTS),weights_sha256=file_hash(WEIGHTS),
                      repository=str(REPOSITORY),revision=revision.stdout.strip(),
                      architecture="DINOv2 ViT-B/14, four registers",strict_load=True,frozen=True,
                      image_size=224,normalization_mean=mean.flatten().tolist(),
                      normalization_std=std.flatten().tolist(),extraction_seconds=time.perf_counter()-started,
                      images=[dict(record,path=f"images/card_{i:03d}.png",sha256=file_hash(images / f"card_{i:03d}.png"))
                              for i,record in enumerate(records)])
    (output / "feature_provenance.json").write_text(json.dumps(provenance,indent=2))
    return features,records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",type=Path,help="Reuse extracted features instead of loading the encoder")
    parser.add_argument("--output",type=Path,default=ROOT / "run")
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0,str(RECOVERY / "composition"))
    from experiment import meanings,heldout_mask,metrics
    output = args.output
    output.mkdir(exist_ok=True,parents=True)
    if (output / "results.json").exists():
        raise FileExistsError("Preserve the completed image-feature experiment")
    if args.features:
        values = np.load(args.features,allow_pickle=False)["features"]
        records = [dict(state=state.tolist(),style=style) for style in range(8) for state in STATES]
        if values.shape != (144,768) or not np.isfinite(values).all():
            raise ValueError("Unexpected saved image features")
    else:
        values,records = extract(output)
    labels = np.array([r["state"] for r in records])
    styles = np.array([r["style"] for r in records])
    excluded = (labels[:,0] == 0)&(labels[:,1] == 0)
    train = np.flatnonzero((styles < 4)&~excluded)
    validation = np.flatnonzero((styles >= 4)&(styles < 6)&~excluded)
    test = np.flatnonzero(styles >= 6)
    assert len(train) == 64 and len(validation) == 32 and len(test) == 36
    scaler = StandardScaler().fit(values[train])
    transformed = scaler.transform(values)
    logistic = [LogisticRegression(C=1.0,max_iter=2000,random_state=20261023).fit(transformed[train],labels[train,i]) for i in range(3)]
    levels = (np.array([-1.,0.,1.]),np.array([-1.,0.,1.]),np.array([-1.,1.]))
    coordinate_targets = np.stack([level[labels[:,i]] for i,level in enumerate(levels)],axis=1)
    ridge = Ridge(alpha=1.0).fit(transformed[train],coordinate_targets[train])
    predictions = {"logistic":np.stack([model.predict(transformed) for model in logistic],axis=1)}
    coordinates = ridge.predict(transformed)
    predictions["semantic_grid"] = np.stack([np.abs(coordinates[:,i,None]-level).argmin(axis=1) for i,level in enumerate(levels)],axis=1)
    validation_scores = {name:float((predicted[validation] == labels[validation]).all(axis=1).mean()) for name,predicted in predictions.items()}
    selected = max(validation_scores,key=validation_scores.get)
    lookup = {(tuple(r["state"]),r["style"]):i for i,r in enumerate(records)}
    scene_meanings = meanings()
    heldout = heldout_mask(scene_meanings)
    evaluations = {}
    for name,predicted in predictions.items():
        result = dict(validation_object_exact=validation_scores[name],
                      test_object_exact=float((predicted[test] == labels[test]).all(axis=1).mean()),
                      test_field_accuracy=(predicted[test] == labels[test]).mean(axis=0).tolist())
        for tag,scene_subset in (("heldout",scene_meanings[heldout]),("familiar",scene_meanings[~heldout])):
            expected,got = [],[]
            for scene in scene_subset:
                for first_style,second_style in product((6,7),repeat=2):
                    indices = [lookup[(tuple(scene[0]),first_style)],lookup[(tuple(scene[1]),second_style)]]
                    expected.append(scene)
                    got.append(predicted[indices])
            result[tag] = metrics(np.array(got),np.array(expected))
        clean_expected = scene_meanings[heldout]
        got = np.array([[predicted[lookup[(tuple(obj),6)]] for obj in scene] for scene in clean_expected])
        permutation = np.random.default_rng(20261023).permutation(len(got))
        result["mismatched"] = metrics(got[permutation],clean_expected)
        changed = clean_expected.copy()
        changed[:,0,0] = (changed[:,0,0]+1)%3
        changed_predictions = np.array([[predicted[lookup[(tuple(obj),6)]] for obj in scene] for scene in changed])
        result["color_counterfactual"] = metrics(changed_predictions,changed)
        mask = np.ones((2,3),dtype=bool)
        mask[0,0] = False
        result["unrelated_fields_unchanged"] = float((changed_predictions[:,mask] == got[:,mask]).all(axis=1).mean())
        evaluations[name] = result
    report = dict(encoder="frozen cached DINOv2 ViT-B/14 reg4",selected_by_validation=selected,
                  validation_scores=validation_scores,train_indices=train.tolist(),validation_indices=validation.tolist(),
                  test_indices=test.tolist(),evaluations=evaluations,
                  note="Held-out combinations concern compiler training, not unknown pretrained-encoder exposure. Supplied object cards and roles; rendered images.")
    (output / "results.json").write_text(json.dumps(report,indent=2))
    model = dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                 logistic=[dict(classes=m.classes_.tolist(),coefficients=m.coef_.tolist(),intercept=m.intercept_.tolist()) for m in logistic],
                 semantic_grid=dict(coefficients=ridge.coef_.tolist(),intercept=ridge.intercept_.tolist(),levels=[a.tolist() for a in levels]),
                 selected=selected,input_dim=768,encoder=report["encoder"])
    (output / "readout.json").write_text(json.dumps(model,indent=2))
    print(json.dumps(dict(selected=selected,validation_scores=validation_scores,
                          results={name:dict(heldout_program_accuracy=r["heldout"]["exact_accuracy"],
                                            familiar_program_accuracy=r["familiar"]["exact_accuracy"],
                                            test_field_accuracy=r["test_field_accuracy"]) for name,r in evaluations.items()}),indent=2),flush=True)


if __name__ == "__main__":
    main()
