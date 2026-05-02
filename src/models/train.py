# src/models/train.py

import warnings
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data.preprocess import build_preprocessor


# ─────────────────────────────────────────────────
# 1. MODELS + HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────────

def get_models_and_params():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, solver="saga"),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": ["balanced", None],
            }
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 15],
                "model__min_samples_leaf": [1, 2, 5],
                "model__class_weight": ["balanced", "balanced_subsample"],
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0],
                "model__min_samples_leaf": [1, 2, 5],
            }
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 1.0],
            }
        ),
        "SVC": (
            SVC(probability=True, random_state=42),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": ["balanced", None],
            }
        ),
    }


# ─────────────────────────────────────────────────
# 2. BUILD PIPELINE CÓ SMOTE
# ─────────────────────────────────────────────────

def build_pipeline(model, use_smote: bool = True):
    preprocessor = build_preprocessor()
    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append((
            "smote",
            SMOTE(
                sampling_strategy=0.5,
                random_state=42,
                k_neighbors=5
            )
        ))

    steps.append(("model", model))
    return ImbPipeline(steps)


# ─────────────────────────────────────────────────
# 3. TÌM THRESHOLD TỐI ƯU
# ─────────────────────────────────────────────────

def find_optimal_threshold(model, X_val, y_val, beta: float = 2.0):
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    f_beta = (
        (1 + beta ** 2) * precisions * recalls
        / (beta ** 2 * precisions + recalls + 1e-9)
    )

    best_idx = np.argmax(f_beta)
    best_threshold = (
        float(thresholds[best_idx])
        if best_idx < len(thresholds)
        else 0.5
    )

    return best_threshold


# ─────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test,
                   threshold: float = 0.5,
                   best_name: str = "",
                   best_cv_auc: float = 0.0):

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\nBest Model: {best_name} | AUC: {best_cv_auc:.4f}")
    print(classification_report(y_test, y_pred))
    print(cm)
    print(f"ROC-AUC: {auc}")

    return {"auc": auc}


# ─────────────────────────────────────────────────
# 5. HÀM CHÍNH
# ─────────────────────────────────────────────────

def train_and_select_model(
        X_train, y_train,
        X_val=None, y_val=None,
        use_smote: bool = True,
        n_iter_search: int = 20
):
    models_and_params = get_models_and_params()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_cv_score = 0
    best_pipeline = None
    best_name     = None

    for name, (model, param_grid) in models_and_params.items():

        pipeline = build_pipeline(model, use_smote=use_smote)

        searcher = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter_search,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            refit=True,
            verbose=0,
            error_score=0.0         # ← tránh crash nếu 1 combo params lỗi
        )

        searcher.fit(X_train, y_train)

        print(f"{name} AUC: {searcher.best_score_:.4f}")

        if searcher.best_score_ > best_cv_score:
            best_cv_score = searcher.best_score_
            best_pipeline = searcher.best_estimator_
            best_name     = name

    if X_val is not None and y_val is not None:
        optimal_threshold = find_optimal_threshold(
            best_pipeline, X_val, y_val, beta=2.0
        )
    else:
        optimal_threshold = 0.5

    return best_pipeline, best_name, best_cv_score, optimal_threshold