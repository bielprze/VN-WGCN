# Application-of-Voronoi-Neighborhood-Weighted-Graph-Convolutional-Networks

This repository contains end-to-end code for forecasting traffic speeds using:

- **ST-GCN / VNWS-TGCN** (Spatio-Temporal Graph Convolutional Networks)  
  – `train.py` & `evaluate.py`  
- **Simple LSTM** baseline  
  – `simple_lstm.py` & `evaluate_simple_lstm.py`  
- **Helper modules**  
  – `helper.py` & `gcnnmodel.py`  
- **R analysis scripts**  
  – `evaluate_results.R`  

All scripts are PEP 8-compliant, fully documented, and use consistent configuration blocks.

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
├── train.py                      # Train ST-GCN / VNWS-TGCN
├── evaluate.py                   # Evaluate ST-GCN models
├── simple_lstm.py                # Train simple LSTM baseline
├── evaluate_simple_lstm.py       # Evaluate LSTM baseline
├── helper.py                     # GeoJSON & CSV loading + preprocessing
├── gcnnmodel.py                  # GraphConv & LSTMGC model definitions
├── evaluate_results.R            # R script for summarising & comparing
├── requirements.txt              # Python dependencies
└── 2024-03-01_35/                # Folder of sensor CSVs (time series)
    └── *.csv
```

---

## ⚙️ Configuration

At the top of each Python script you’ll find a **config block**:

```python
# Paths
DATA_FOLDER = Path("2024-03-01_35")
GEOJSON_PATH = Path("sensors_location.geojson")

# Splits
TRAIN_SIZE = 0.5
VAL_SIZE   = 0.2

# Model settings
INPUT_SEQ_LEN      = 12
FORECAST_HORIZONS  = [1,2,3]
LSTM_UNITS         = 200
GRAPH_CONV_PARAMS  = { "aggregation_type":"mean", ... }
# …
```

Modify these constants to adapt to your data or experiments.

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
