"""Reproduce the grouped proxy baselines from the bundled feature benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT / "benchmarks/proxy_features.npz")
    args = parser.parse_args()
    data = np.load(args.data,allow_pickle=False)
    raw,y,groups,domains = (data[k] for k in ("features","labels","groups","domains"))
    normalized = (raw-raw.mean(axis=(1,2),keepdims=True))/(raw.std(axis=(1,2),keepdims=True)+1e-8)
    values = np.concatenate([normalized.reshape(len(raw),-1),np.eye(2)[domains]],axis=1)
    group_y = np.array([y[groups == i][0] for i in range(68)])
    examples = {"logistic":[],"rbf_svm":[]}
    for fold,(trainval,test) in enumerate(StratifiedKFold(n_splits=4,shuffle=True,random_state=20261022).split(range(68),group_y)):
        train,validation = train_test_split(trainval,test_size=.20,stratify=group_y[trainval],random_state=20261021)
        train_rows = np.flatnonzero(np.isin(groups,train))
        test_rows = np.flatnonzero(np.isin(groups,test))
        models = {"logistic":LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=20261021),
                  "rbf_svm":SVC(C=10,gamma="scale",class_weight="balanced")}
        for name,model in models.items():
            estimator = make_pipeline(StandardScaler(),model).fit(values[train_rows],y[train_rows])
            predicted = estimator.predict(values[test_rows])
            for group in sorted(test):
                vote = Counter(predicted[groups[test_rows] == group]).most_common(1)[0][0]
                examples[name].append(dict(group=int(group),fold=fold,correct=bool(vote == group_y[group])))
    report = {name:dict(correct=sum(row["correct"] for row in rows),total=len(rows),
                       accuracy=sum(row["correct"] for row in rows)/len(rows)) for name,rows in examples.items()}
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
