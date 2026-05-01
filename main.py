from src.data.load_data import load_data
from src.features.feature_engineering import build_features
from src.models.train import train_and_select_model
from src.models.evaluate import evaluate
from sklearn.model_selection import train_test_split
import joblib

df = load_data("data/raw/customer_data.csv")
df = build_features(df)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model, name, score = train_and_select_model(X_train, y_train)

evaluate(model, X_test, y_test)

joblib.dump(model, "models/best_model.pkl")
