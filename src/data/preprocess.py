from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor():

    numeric = [
        "credit_score", "age", "tenure",
        "balance", "estimated_salary",
        "balance_per_product"
    ]

    categorical = ["country", "gender"]

    binary = ["credit_card", "active_member", "is_senior", "inactive"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("bin", "passthrough", binary)
    ])

    return preprocessor