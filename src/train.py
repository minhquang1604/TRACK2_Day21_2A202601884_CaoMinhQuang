import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu (tu README, do tren toan bo du lieu Adult
# sau khi lam sach). Neu ty le tren tap train lech qua 5 diem phan tram so voi
# con so nay, co the du lieu dang bi lech (data drift) va can kiem tra lai.
REFERENCE_POSITIVE_RATIO = 0.248
DRIFT_TOLERANCE = 0.05

# Bonus 2: cac nguong duoc quet de tim f1 toi uu, thay vi chi dung nguong
# mac dinh 0.5 cua model.predict().
THRESHOLD_SWEEP = np.arange(0.1, 0.91, 0.05)


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout,
                    tinh o nguong mac dinh 0.5 (dung lam can cu quality gate).
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: canh bao lech phan phoi du lieu truoc khi huan luyen
    positive_ratio = float(y_train.mean())
    drift = abs(positive_ratio - REFERENCE_POSITIVE_RATIO)
    drift_warning = None
    if drift > DRIFT_TOLERANCE:
        drift_warning = (
            f"CANH BAO DATA DRIFT: ty le lop duong trong tap train la "
            f"{positive_ratio:.1%}, lech {drift:.1%} so voi ty le tham chieu "
            f"{REFERENCE_POSITIVE_RATIO:.1%} (nguong cho phep {DRIFT_TOLERANCE:.0%})."
        )
        print(drift_warning)
    else:
        print(
            f"Ty le lop duong trong tap train: {positive_ratio:.1%} "
            f"(tham chieu {REFERENCE_POSITIVE_RATIO:.1%}) - trong nguong cho phep."
        )

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout va tinh chi so o nguong mac dinh 0.5
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_ratio", positive_ratio)

        # Bonus 2: quet nguong tu 0.1 den 0.9 (buoc 0.05) tren xac suat du doan
        # de tim nguong cho f1 cao nhat, thay vi chi dung nguong mac dinh 0.5.
        probs = model.predict_proba(X_eval)[:, 1]
        best_threshold, best_threshold_f1 = 0.5, f1
        for t in THRESHOLD_SWEEP:
            preds_t = (probs >= t).astype(int)
            f1_t = f1_score(y_eval, preds_t)
            if f1_t > best_threshold_f1:
                best_threshold, best_threshold_f1 = round(float(t), 2), f1_t
        print(
            f"Nguong toi uu: {best_threshold:.2f} (f1={best_threshold_f1:.4f}) "
            f"so voi nguong mac dinh 0.5 (f1={f1:.4f})"
        )
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)

        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2. Quality gate van dung
        # dung "f1_score" (nguong mac dinh 0.5) de quyet dinh trien khai.
        os.makedirs("outputs", exist_ok=True)
        report = {
            "f1_score": f1,
            "accuracy": acc,
            "positive_ratio": positive_ratio,
            "best_threshold": best_threshold,
            "best_threshold_f1": best_threshold_f1,
        }
        if drift_warning:
            report["data_drift_warning"] = drift_warning
        with open("outputs/report.json", "w") as f_out:
            json.dump(report, f_out)

        # Bonus 3: bao cao chi tiet precision/recall tung lop + confusion matrix,
        # de CI/CD co the luu lai va nguoi doc hieu ro mo hinh sai o dau.
        tn, fp, fn, tp = confusion_matrix(y_eval, preds, labels=[0, 1]).ravel()
        detail_lines = [
            "=== Confusion Matrix (nguong mac dinh 0.5) ===",
            "                   du_doan_thap   du_doan_cao",
            f"thuc_te_thap       {tn:<14}{fp}",
            f"thuc_te_cao        {fn:<14}{tp}",
            "",
            "=== Precision / Recall theo lop ===",
            classification_report(
                y_eval, preds, labels=[0, 1],
                target_names=["thu_nhap_thap", "thu_nhap_cao"],
            ),
            "=== Nhan xet chi phi sai lam ===",
            "False Negative (FN = bo sot nguoi thu nhap cao, recall lop duong thap)",
            "so voi False Positive (FP = gan nham nguoi thu nhap thap thanh cao,",
            "precision lop duong thap): voi bai toan sang loc/uu dai tin dung, FP",
            "ton kem hon vi he thong cap uu dai/han muc cho nguoi khong du dieu",
            "kien se gay thiet hai tai chinh truc tiep, trong khi FN chi la bo lo",
            "co hoi tiep can mot khach hang tiem nang (chi phi co hoi, khong phai",
            "thiet hai truc tiep). Vi vay khi can danh doi, uu tien precision cao",
            "hon cho lop thu nhap cao.",
        ]
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w") as f_out:
            f_out.write("\n".join(detail_lines) + "\n")
        print("\n".join(detail_lines))

        # Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # Tra ve f1 o nguong mac dinh 0.5 (dung lam can cu quality gate o Buoc 2)
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
