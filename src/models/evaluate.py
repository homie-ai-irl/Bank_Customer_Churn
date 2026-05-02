# src/models/evaluate.py

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────
# 1. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────

def plot_feature_importance(best_model, save_dir: str = "reports"):
    """
    Vẽ Top 10 feature quan trọng nhất từ GradientBoosting.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Lấy model và preprocessor từ pipeline
    model       = best_model.named_steps["model"]
    preprocessor = best_model.named_steps["preprocessor"]

    # Lấy tên feature sau khi transform
    feature_names = preprocessor.get_feature_names_out()

    # Làm sạch tên feature cho dễ đọc
    clean_names = []
    for name in feature_names:
        name = name.replace("num__", "").replace("ord__", "") \
                   .replace("bin__", "").replace("cat__", "")
        clean_names.append(name)

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]

    top_names  = [clean_names[i] for i in indices][::-1]
    top_values = importances[indices][::-1]

    # Màu sắc theo mức độ quan trọng
    colors = ["#d32f2f" if v >= 0.1 else
              "#f57c00" if v >= 0.05 else
              "#388e3c"
              for v in top_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_names, top_values, color=colors, edgecolor="white", height=0.6)

    # Thêm giá trị vào cuối mỗi bar
    for bar, val in zip(bars, top_values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, color="#333333")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#d32f2f", label="Rất quan trọng  (≥ 10%)"),
        mpatches.Patch(color="#f57c00", label="Quan trọng      (5–10%)"),
        mpatches.Patch(color="#388e3c", label="Bình thường     (< 5%)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("Top 10 Features ảnh hưởng đến Churn", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(top_values) * 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{save_dir}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()

    # In bảng top 10
    print("\n  Top 10 Features:")
    print(f"  {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 47)
    for name, val in zip(top_names[::-1], top_values[::-1]):
        bar = "█" * int(val * 100)
        print(f"  {name:<35} {val:>8.4f}  {bar}")


# ─────────────────────────────────────────────────
# 2. BUSINESS RECOMMENDATION
# ─────────────────────────────────────────────────

def print_business_recommendation(X_test: pd.DataFrame, y_test: pd.Series,
                                   best_model, threshold: float = 0.5):
    """
    Phân tích đặc điểm nhóm khách hàng có nguy cơ churn cao
    và đưa ra đề xuất kinh doanh cụ thể.
    """

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    df = X_test.copy()
    df["churn_actual"]   = y_test.values
    df["churn_predicted"] = y_pred
    df["churn_proba"]    = y_proba

    # Nhóm predicted churn
    churned = df[df["churn_predicted"] == 1]
    retained = df[df["churn_predicted"] == 0]

    sep = "=" * 56

    print(f"\n{sep}")
    print("   BUSINESS RECOMMENDATION")
    print(sep)

    print(f"\n  Tong so khach hang co nguy co churn : {len(churned):,}")
    print(f"  Tong so khach hang an toan          : {len(retained):,}")

    # ── Insight 1: Tuổi ───────────────────────────
    print(f"\n{'─' * 56}")
    print("  [1] PHAN TICH THEO NHOM TUOI")
    print(f"{'─' * 56}")
    if "age_group" in df.columns:
        age_churn = df.groupby("age_group")["churn_predicted"].mean().sort_values(ascending=False)
        for group, rate in age_churn.items():
            bar = "█" * int(rate * 20)
            print(f"  {group:<10} churn rate: {rate:.1%}  {bar}")
        top_age = age_churn.index[0]
        print(f"\n  → Nhom tuoi '{top_age}' co ty le churn cao nhat")
        print(f"  ĐE XUAT: Trien khai chuong trinh uu dai dac biet")
        print(f"           (giam phi, tang lai suat) cho nhom '{top_age}'")

    # ── Insight 2: Balance ────────────────────────
    print(f"\n{'─' * 56}")
    print("  [2] PHAN TICH THEO BALANCE")
    print(f"{'─' * 56}")
    if "is_zero_balance" in df.columns:
        zero_bal_churn    = df[df["is_zero_balance"] == 1]["churn_predicted"].mean()
        nonzero_bal_churn = df[df["is_zero_balance"] == 0]["churn_predicted"].mean()
        print(f"  Khach balance = 0  : churn rate {zero_bal_churn:.1%}")
        print(f"  Khach balance > 0  : churn rate {nonzero_bal_churn:.1%}")
        print(f"\n  → Khach balance = 0 co nguy co churn cao hon "
              f"{zero_bal_churn / (nonzero_bal_churn + 1e-9):.1f}x")
        print(f"  ĐE XUAT: Gui thong bao khuyen khich gui tiet kiem,")
        print(f"           tang lai suat uu dai cho lan gui dau tien")

    # ── Insight 3: Số sản phẩm ───────────────────
    print(f"\n{'─' * 56}")
    print("  [3] PHAN TICH THEO SO SAN PHAM SU DUNG")
    print(f"{'─' * 56}")
    if "products_number" in df.columns:
        prod_churn = df.groupby("products_number")["churn_predicted"].mean().sort_values(ascending=False)
        for prod, rate in prod_churn.items():
            bar = "█" * int(rate * 20)
            print(f"  {prod} san pham  churn rate: {rate:.1%}  {bar}")
        print(f"\n  ĐE XUAT: Khach dung 1 san pham → Cross-sell sang")
        print(f"           the tin dung, bao hiem, quy dau tu")

    # ── Insight 4: Active member ──────────────────
    print(f"\n{'─' * 56}")
    print("  [4] PHAN TICH THEO HOAT DONG TAI KHOAN")
    print(f"{'─' * 56}")
    if "active_member" in df.columns:
        active_churn   = df[df["active_member"] == 1]["churn_predicted"].mean()
        inactive_churn = df[df["active_member"] == 0]["churn_predicted"].mean()
        print(f"  Khach active      : churn rate {active_churn:.1%}")
        print(f"  Khach not active  : churn rate {inactive_churn:.1%}")
        print(f"\n  → Khach khong hoat dong co nguy co churn cao hon "
              f"{inactive_churn / (active_churn + 1e-9):.1f}x")
        print(f"  ĐE XUAT: Gui email/SMS khuyen khich su dung app,")
        print(f"           tang diem thuong cho giao dich hang thang")

    # ── Insight 5: Quốc gia ───────────────────────
    print(f"\n{'─' * 56}")
    print("  [5] PHAN TICH THEO QUOC GIA")
    print(f"{'─' * 56}")
    if "country" in df.columns:
        country_churn = df.groupby("country")["churn_predicted"].mean().sort_values(ascending=False)
        for country, rate in country_churn.items():
            bar = "█" * int(rate * 20)
            print(f"  {country:<12} churn rate: {rate:.1%}  {bar}")
        top_country = country_churn.index[0]
        print(f"\n  → '{top_country}' co ty le churn cao nhat")
        print(f"  ĐE XUAT: Nghien cuu nguyen nhan dac thu tai '{top_country}',")
        print(f"           co the do canh tranh phi tu ngan hang khac")

    # ── Insight 6: Credit group ───────────────────
    print(f"\n{'─' * 56}")
    print("  [6] PHAN TICH THEO CREDIT SCORE")
    print(f"{'─' * 56}")
    if "credit_group" in df.columns:
        credit_churn = df.groupby("credit_group")["churn_predicted"].mean().sort_values(ascending=False)
        for group, rate in credit_churn.items():
            bar = "█" * int(rate * 20)
            print(f"  {group:<15} churn rate: {rate:.1%}  {bar}")
        print(f"\n  ĐE XUAT: Khach credit thap → Ho tro nang cao tin dung")
        print(f"           Khach credit cao   → Giu chan bang goi VIP/Premium")

    # ── Tổng kết ──────────────────────────────────
    print(f"\n{sep}")
    print("  TONG KET DE XUAT CHIEN LUOC")
    print(sep)
    print("""
  Ngan hang nen trien khai he thong canh bao som:

  Nguy co THAP  (proba < 0.4) → Theo doi dinh ky hang quy
  Nguy co TRUNG (0.4 – 0.6)   → Gui khuyen mai, uu dai phi
  Nguy co CAO   (> 0.6)        → Goi dien truc tiep, retention offer

  Muc tieu: Giu chan it nhat 60% khach co nguy co cao
  Uoc tinh: Moi 1% giam churn = tang doanh thu ~X ty dong
    """)
    print(sep)