from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np

from src.data.preprocess import build_preprocessor


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }


def train_and_select_model(X_train, y_train):

    preprocessor = build_preprocessor()
    models = get_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_score = 0
    best_model = None
    best_name = None

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        mean_score = np.mean(scores)
        print(f"{name} AUC: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe
            best_name = name

    print(f"\nBest Model: {best_name}")

    best_model.fit(X_train, y_train)

    return best_model, best_name, best_score