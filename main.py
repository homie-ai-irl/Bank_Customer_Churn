# main.py

import os
import warnings
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from src.data.preprocess import load_data
from src.features.feature_engineering import add_features
from src.models.train import train_and_select_model, evaluate_model
from src.models.evaluate import plot_feature_importance, print_business_recommendation


def main():
    print("=" * 56)
    print("   CHURN PREDICTION PIPELINE")
    print("=" * 56)

    print("\n[1] Loading data...")
    X, y = load_data("data/raw/customer_data.csv")

    print("\n[2] Feature engineering...")
    X = add_features(X)
    print(f"  Features sau khi them: {X.shape[1]} cot")

    print("\n[3] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    print("\n[4] Training models...")
    best_model, best_name, best_auc, threshold = train_and_select_model(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        use_smote=True,
        n_iter_search=20
    )

    print("\n[5] Final evaluation on test set...")
    evaluate_model(
        best_model, X_test, y_test,
        threshold=threshold,
        best_name=best_name,
        best_cv_auc=best_auc
    )

    print("\n[6] Feature Importance...")
    plot_feature_importance(best_model, save_dir="reports")

    print("\n[7] Business Recommendation...")
    print_business_recommendation(X_test, y_test, best_model, threshold=threshold)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{best_name}_best.pkl"
    joblib.dump({"model": best_model, "threshold": threshold}, save_path)


if __name__ == "__main__":
    main()