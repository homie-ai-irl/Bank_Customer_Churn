def build_features(df):
    df = df.copy()

    df["balance_per_product"] = df["balance"] / (df["products_number"] + 1)
    df["is_senior"] = (df["age"] > 45).astype(int)
    df["inactive"] = (df["active_member"] == 0).astype(int)

    return df