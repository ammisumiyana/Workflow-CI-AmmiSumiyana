"""
modelling.py — Workflow CI (Kriteria 3)
Dijalankan oleh MLflow Project / GitHub Actions.
Menerima hyperparameter via CLI.
Tracking online ke DagsHub.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

# ── Tracking DagsHub via env var (di-set oleh GitHub Actions) ──
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

# ── CLI Parameters ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',      type=int, default=100)
parser.add_argument('--max_depth',         type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf',  type=int, default=1)
args = parser.parse_args()

n_estimators      = args.n_estimators
max_depth         = args.max_depth
min_samples_split = args.min_samples_split
min_samples_leaf  = args.min_samples_leaf

# Nilai 0 untuk max_depth → None (unlimited)
if max_depth == 0:
    max_depth = None

print(f'[CI] n_estimators={n_estimators}, max_depth={max_depth}, '
      f'min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}')

# ── Load Data ──────────────────────────────────────────────────
X_train = pd.read_csv('heartdisease_preprocessing/X_train.csv')
X_test  = pd.read_csv('heartdisease_preprocessing/X_test.csv')
y_train = pd.read_csv('heartdisease_preprocessing/y_train.csv').squeeze()
y_test  = pd.read_csv('heartdisease_preprocessing/y_test.csv').squeeze()

print(f'[CI] Train: {X_train.shape} | Test: {X_test.shape}')

# ── Eksperimen MLflow ──────────────────────────────────────────
mlflow.set_experiment('HeartDisease-CI')

with mlflow.start_run(
    run_name=f'CI-RF-{n_estimators}-{max_depth}-'
             f'{min_samples_split}-{min_samples_leaf}'
) as run:

    # ── Training ──────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Manual Logging Params ─────────────────────────────────
    mlflow.log_param('n_estimators',      n_estimators)
    mlflow.log_param('max_depth',         max_depth)
    mlflow.log_param('min_samples_split', min_samples_split)
    mlflow.log_param('min_samples_leaf',  min_samples_leaf)

    # ── Manual Logging Metrics ────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)

    mlflow.log_metric('accuracy',  acc)
    mlflow.log_metric('f1_score',  f1)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall',    recall)
    mlflow.log_metric('roc_auc',   auc)

    # ── Artifact: Confusion Matrix ────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold')
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=['Tdk Sakit', 'Sakit'],
        yticklabels=['Tdk Sakit', 'Sakit'],
        title='Confusion Matrix CI'
    )
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')

    # ── Log Model ─────────────────────────────────────────────
    mlflow.sklearn.log_model(model, 'model')

    # ── Simpan run_id ke file ─────────────────────────────────
    run_id = run.info.run_id
    with open('latest_run_id.txt', 'w') as f:
        f.write(run_id)

    # ── Ringkasan ─────────────────────────────────────────────
    print(f'\n[CI] Run ID   : {run_id}')
    print(f'[CI] Accuracy : {acc:.4f}')
    print(f'[CI] F1 Score : {f1:.4f}')
    print(f'[CI] ROC AUC  : {auc:.4f}')
