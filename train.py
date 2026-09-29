"""Обучение модели прогноза дефолта по кредиту (Lending Club).

Запуск:
    python train.py accepted_2007_to_2018Q2.csv
    python train.py accepted_2007_to_2018Q2.csv --nrows 300000   # быстрее
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Только признаки, известные на момент подачи заявки (без утечки данных)
NUM = ["loan_amnt", "term", "int_rate", "installment", "emp_length",
       "annual_inc", "dti", "delinq_2yrs", "fico", "open_acc", "pub_rec",
       "revol_bal", "revol_util", "total_acc", "mort_acc",
       "pub_rec_bankruptcies", "credit_history_years"]
CAT = ["grade", "home_ownership", "verification_status", "purpose"]
FEATURES = NUM + CAT

RAW_COLS = ["loan_status", "loan_amnt", "term", "int_rate", "installment",
            "grade", "emp_length", "home_ownership", "annual_inc",
            "verification_status", "purpose", "dti", "delinq_2yrs",
            "earliest_cr_line", "issue_d", "fico_range_low", "fico_range_high",
            "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "mort_acc", "pub_rec_bankruptcies"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["target"] = (df["loan_status"] != "Fully Paid").astype(int)
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["emp_length"] = (df["emp_length"].astype(str)
                        .str.replace("< 1", "0").str.extract(r"(\d+)")[0].astype(float))
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    start = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (issue - start).dt.days / 365.25
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    print("Читаю данные...")
    df = pd.read_csv(args.csv, usecols=RAW_COLS, nrows=args.nrows, low_memory=False)
    df = prepare(df)
    print(f"Строк после фильтрации: {len(df):,} | доля дефолтов: {df['target'].mean():.1%}")

    X, y = df[FEATURES], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=np.nan), CAT)],
        remainder="passthrough")
    # после трансформера категориальные признаки идут первыми
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        categorical_features=list(range(len(CAT))), random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    print("Обучаю модель...")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {"roc_auc": roc_auc_score(y_te, proba),
               "pr_auc": average_precision_score(y_te, proba),
               "base_rate": float(y_te.mean()), "n_train": len(X_tr)}
    print(f"ROC-AUC: {metrics['roc_auc']:.3f} | PR-AUC: {metrics['pr_auc']:.3f}")

    print("Считаю важность признаков...")
    idx = X_te.sample(min(5000, len(X_te)), random_state=1).index
    imp = permutation_importance(pipe, X_te.loc[idx], y_te.loc[idx],
                                 scoring="roc_auc", n_repeats=3, random_state=1)
    importance = (pd.Series(imp.importances_mean, index=FEATURES)
                  .sort_values(ascending=False).head(12).to_dict())

    options = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CAT}
    defaults = {c: float(df[c].median()) for c in NUM}
    joblib.dump({"pipeline": pipe, "metrics": metrics, "importance": importance,
                 "options": options, "defaults": defaults}, args.out)
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
