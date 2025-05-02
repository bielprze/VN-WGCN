# Application-of-Voronoi-Neighborhood-Weighted-Graph-Convolutional-Networks

This repository contains end-to-end code for forecasting traffic speeds using:

- **ST-GCN / VNWS-TGCN** (Spatio-Temporal Graph Convolutional Networks)  
  – `train_GNN.py` & `evaluate_GNN.py`  
- **Simple LSTM** baseline  
  – `simple_lstm.py` & `evaluate_simple_lstm.py`  
- **Helper modules**  
  – `helper.py` & `gcnnmodel.py`  
- **R analysis scripts**  
  – `evaluate_results.R`  

All scripts are PEP 8-compliant, fully documented, and use consistent configuration blocks.

---

## 📊 Data

Traffic data originate from Darmstadt, Germany. The `2024-03-01_35/` folder is named after its start date (March 1, 2024) and contains 35 days of 10-minute interval recordings. Each CSV file corresponds to a single crossroads (named by sensor ID, e.g., `A003.csv`, `A007.csv`, `A024.csv`, `A113.csv`, etc.). Inside each file, rows represent 10-minute timestamps, with columns for each local sensor’s vehicle count at that crossroads and a `Total` column summing all sensors over each interval.

---

## 🚀 Quickstart

1. **Clone this repo**  
   ```bash
   git clone https://github.com/bielprze/Application-of-Voronoi-Neighborhood-Weighted-Graph-Convolutional-Networks.git
   cd Application-of-Voronoi-Neighborhood-Weighted-Graph-Convolutional-Networks
   ```

2. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install R dependencies**  
   In an R session:
   ```r
   install.packages("tidyverse", repos = "https://cloud.r-project.org/")
   ```

4. **Fetch / prepare your data**  
   Place your CSV time-series files under `2024-03-01_35/` and your GeoJSON at `sensors_location.geojson`.

5. **Train ST-GCN models**  
   ```bash
   python train_GNN.py
   ```
   - Checkpoints go into `weights/`  
   - Progress bars via Keras (`verbose=1`) and `tqdm` on outer loops.

6. **Evaluate ST-GCN models**  
   ```bash
   python evaluate_GNN.py
   ```
   - Aggregated metrics in `evaluation_results.csv`

7. **Train simple LSTM baseline**  
   ```bash
   python simple_lstm.py
   ```
   - Checkpoints in `weights_lstm/`

8. **Evaluate LSTM baseline**  
   ```bash
   python evaluate_simple_lstm.py
   ```
   - Aggregated metrics in `evaluation_simple_lstm.csv`

9. **Summarise & compare results in R**  
   ```bash
   Rscript evaluate_results.R
   ```
   - Generates CSV summaries:
     - `summary_stgcn_by_horizon.csv`  
     - `summary_stgcn_by_horizon_scaler.csv`  
     - `summary_lstm_by_horizon.csv`  
     - `summary_combined_by_horizon.csv`

---

## 📁 Repository Structure

```
.
├── train_GNN.py                  # Train ST-GCN / VNWS-TGCN
├── evaluate_GNN.py               # Evaluate ST-GCN models
├── simple_lstm.py                # Train simple LSTM baseline
├── evaluate_simple_lstm.py       # Evaluate LSTM baseline
├── helper.py                     # GeoJSON & CSV loading + preprocessing
├── gcnnmodel.py                  # GraphConv & LSTMGC model definitions
├── evaluate_results.R            # R script for summarising & comparing
├── config.yaml                   # configuration for evaluate_GNN.py and train_GNN.py
├── requirements.txt              # Python dependencies
└── 2024-03-01_35/                # Folder of sensor CSVs (time series)
    └── *.csv
```

---

## ⚙️ Configuration

All configurable parameters (paths, hyperparameters, data splits, etc.) are centralized in `config.yaml`. Edit that file to adjust settings:

```yaml
# Example entries in config.yaml

# Data & paths
folder_to_load: "2024-03-01_35/"
geojson_path: "sensors_location.geojson"
save_folder: "weights"
output_csv: "evaluation_results.csv"

# Train/val/test split
train_size: 0.5
val_size: 0.2

# Graph parameters
sigma2: 0.1
epsilon: 0.95

# Delaunay settings
use_delaunay: true
max_depth: 5
adj_scaler_options: [0, 1, 2]

# Model / data parameters
in_feat: 1
out_feat: 10
lstm_units: 64
graph_conv_params:
  aggregation_type: "mean"
  combination_type: "concat"
  activation: null

# Training / evaluation settings
epochs: 40
batch_size: 64
input_seq_len: 12
forecast_horizons: [1, 2, 3]
seeds: [0,1,2,3,4,5,6,7,8,9]
```

---

## 🔧 Dependencies

```txt
tensorflow>=2.11.0
numpy>=1.23.5
pandas>=1.4.2
pyproj>=3.4.1
tqdm>=4.64.1
keras-tqdm>=0.1.0
scipy>=1.7.0
scikit-learn>=1.0
```
