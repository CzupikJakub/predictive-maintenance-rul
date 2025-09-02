# Predictive Maintenance (RUL) – NASA C-MAPSS (FD001)

**Goal**: Build a clean, reproducible baseline for **Remaining Useful Life (RUL)** prediction on rotating equipment using **NASA C-MAPSS**. The notebook is **Colab-ready**, performs **time-series feature engineering**, trains a **baseline ML model**, and saves **artifacts**.

## Why it matters 
Predictive maintenance enables **condition monitoring** of motors and drives, reducing downtime and improving efficiency. This repo demonstrates an **end-to-end pipeline** applicable to **rotating equipment & drives**.

## Quickstart (Colab)
1. Open `turbofan_rul_quickstart.ipynb` in Google Colab.
2. Upload `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (or use the built-in synthetic data).
3. Set `USE_CMAPSS = True` in the first cell.
4. Run all cells – artifacts saved to `./artifacts/` (`model.joblib`, `metrics.json`). Optional plot in `./figures/`.

## Methods
- **Feature engineering**: rolling mean/std and deltas on sensor time-series.
- **Model**: scikit-learn RandomForest baseline (easy to swap to GBM/LSTM).
- **Metrics**: RMSE and MAE; example plot `figures/rul_scatter.png`.
- **Artifacts**: serialized model (`joblib`) + metrics JSON.

## Results (example)
Baseline RF on engineered features yields reasonable RMSE/MAE on FD001 and a monotonic True vs Predicted RUL scatter.

## Next steps
- Upgrade model (GBM/XGBoost/1D-CNN/LSTM).
- Export to **ONNX** and add **quantization** for edge deployment.
- Run in **Databricks**; add CI/CD via **Azure DevOps**.
- Extend to FD002–FD004 and compare generalization.

## Notes
CMAPSS data is **not** included (size/license). Please download from the official source and keep it local.
