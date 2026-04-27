# Explainable Autoencoder-Based Anomaly Detection for Prescriptive Maintenance Optimization in Data Center Cooling Systems

**Course:** Deep Learning (CS5204) | **Institution:** RV University, School of CSE  
**Author:** Nilanjana Jamindar | **Guide:** Dr. Karthikeyan Periyasami

---

## Abstract

High-performance data centers face thermal anomaly issues due to increased heat densities. Traditional static approaches based on rule sets cannot handle interdependent variables. This paper presents an unsupervised anomaly detection approach based on a basic autoencoder architecture. SHapley Additive exPlanations (SHAP) values are used for feature importance to identify anomalies in sensors. The proposed autoencoder is extended with a prescriptive maintenance module to produce recommendations (e.g., adjusting fan speeds). The autoencoder was trained using real-world sensor data and achieved 97% reconstruction accuracy, outperforming baseline models including Isolation Forest, LSTM-Autoencoder, and One-Class SVM. Future plans include LSTM hybrids for time-series analysis and federated learning for large-scale data centers.

---

## Repository Structure

```
├── README.md                        # Project overview and setup instructions
├── requirements.txt                 # Python dependencies
├── /data
│   └── cold_source_control_dataset.csv   # Sensor telemetry dataset (Kaggle)
├── /notebooks
│   └── Final_Data_Center_DL_Project.ipynb  # Full training, evaluation & deployment notebook
├── /results
│   ├── distributions.png                         # Sensor feature distributions
│   ├── correlation_matrix.png                    # Pearson correlation heatmap
│   ├── reconstruction_accuracy.png               # Visual reconstruction accuracy plot
│   ├── WebApp Screenshot - Anomaly 1.png         # Screenshot showing Anomaly-1 on the WebApp
│   ├── WebApp Screenshot - Anomaly 2.png         # Screenshot showing Anomaly-1 on the WebApp
│   ├── WebApp Screenshot - Anomaly 3.png         # Screenshot showing Anomaly-1 on the WebApp
│   └── WebApp Screenshot - Healthy system.png    # Screenshot chowing Healthy System on the WebApp
├── report.pdf                 # IEEE-format final report (submitted to ASSIC 2026)
└── LICENSE                    # GNU Affero General Public License v3.0
```

---

## Problem Statement

Traditional data center monitoring systems:
- Fail to detect **multi-sensor anomalies** across interdependent variables
- Provide no explanation for detected anomalies ("black box" problem)
- Cannot prescribe targeted corrective actions for operators

This project addresses all three gaps using a **Dense Autoencoder + SHAP** pipeline.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – Data Center Cold Source Control Dataset](https://www.kaggle.com/datasets/programmer3/data-center-cold-source-control-dataset) |
| File | `cold_source_control_dataset.csv` |
| Total Observations | 3,498 |
| Features Used | 8 sensor telemetry streams |
| Train / Test Split | 2,798 / 700 (80/20) |

**Input Features:**
- `Server_Workload(%)` — Primary heat source
- `Inlet_Temperature(°C)` — Cold aisle intake temp
- `Outlet_Temperature(°C)` — Hot aisle exhaust temp
- `Ambient_Temperature(°C)` — Outdoor temperature
- `Cooling_Unit_Power_Consumption(kW)` — Total cooling power draw
- `Chiller_Usage(%)` — Refrigeration unit capacity
- `AHU_Usage(%)` — Air Handling Unit fan capacity
- `Temperature_Deviation(°C)` — Anomaly indicator

---

## Model Architecture

A **symmetric bottleneck Dense Autoencoder** trained on normal operational data:

```
Input (8) → Dense/ReLU (16) → Bottleneck/ReLU (4) → Dense/ReLU (16) → Output/Linear (8)
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam | **Epochs:** 100 | **Batch Size:** 32
- **Anomaly Threshold:** 98th percentile of reconstruction error on test set = `0.0993`

---

## Results

| Model | Anomaly Strength Index | Kurtosis | Stability (Std Dev) | Anomalies Detected | Suitability |
|---|---|---|---|---|---|
| **Dense Autoencoder** | **4.959** | **6.136** | **0.026** | **14** | **Real-time App** |
| Isolation Forest | 3.220 | 1.136 | 0.048 | 14 | Fast Baseline |
| LSTM-Autoencoder | 2.647 | 1.198 | 0.141 | 14 | Historical Analysis |
| One-Class SVM | 1.978 | -0.802 | 0.108 | 20 | Clustered Data |

**Overall Reconstruction Accuracy: 97.34%**

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is integrated via `KernelExplainer` to:
1. Rank sensors by their contribution to the anomaly score
2. Map the top contributing sensor to a physical root cause
3. Generate a prescriptive maintenance action (e.g., *"Inspect rack air filters for hot-aisle containment leaks"*)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Nilaja03/Explainable-Anomaly-Detection-for-Maintenance-Optimization-in-Data-Center-Cooling-Systems.git
cd Explainable-Anomaly-Detection-for-Maintenance-Optimization-in-Data-Center-Cooling-Systems
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Open and run all cells in order:
```bash
jupyter notebook notebooks/Final_Data_Center_DL_Project.ipynb
```

### 4. Launch the Streamlit Dashboard (Optional)
The notebook writes and launches `DCapp.py` via `pyngrok`. To run it locally (add your ngrok API Key first):
```bash
streamlit run DCapp.py
```

---

## Research Paper

This work has been submitted to **ASSIC 2026** (International Conference on Advancements in Smart, Secure and Intelligent Computing) for publication on **IEEE Xplore**.  
Conference venue: KIIT-DU Campus, Bhubaneswar, India (Hybrid Mode).  
Submission portal: https://assic.info/
