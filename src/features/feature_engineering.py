# src/features/feature_engineering.py

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm feature từ data gốc dựa trên domain knowledge
    về hành vi khách hàng ngân hàng.
    """
    df = df.copy()

    # 1. Tổng tài sản trên mỗi sản phẩm
    #    Khách dùng nhiều sản phẩm nhưng balance thấp → dấu hiệu rủi ro
    df["balance_per_product"] = (
        df["balance"] / (df["products_number"] + 1)
    )

    # 2. Khách hàng có balance = 0
    #    Tỷ lệ churn của nhóm này thường khác biệt rõ
    df["is_zero_balance"] = (df["balance"] == 0).astype(int)

    # 3. Tỷ lệ balance / estimated_salary
    #    Phản ánh mức độ phụ thuộc vào ngân hàng
    df["balance_salary_ratio"] = (
        df["balance"] / (df["estimated_salary"] + 1)
    )

    # 4. Nhóm tuổi
    #    Khách lớn tuổi thường có churn rate khác
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    ).astype(str)

    # 5. Khách hàng lâu năm nhưng không active → dấu hiệu churn
    df["tenure_active"] = df["tenure"] * df["active_member"]

    # 6. Credit score group
    df["credit_group"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"]
    ).astype(str)

    return df