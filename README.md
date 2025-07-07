# Predicting Global Trade Hubs: A Hybrid GNN & XAI Framework

This repository contains the complete pipeline for the "Predicting Global Trade Hubs" project, a comprehensive study that leverages dynamic network analysis, time-series modeling, and Graph Neural Network (GNN) inspired features to forecast bilateral trade flows. The project culminates in a deep, explainable AI (XAI) analysis to interpret the behavior of the best-performing model.

The framework is built on a large dataset of UN Comtrade data spanning 37 years (1988-2024) and demonstrates a scalable, end-to-end process from raw data ingestion to advanced model interpretation.

## Key Findings & Highlights

1.  **The Power of the Baseline:** A simple Naive Forecast (predicting this year's trade value as the same as last year's) proved to be an incredibly strong baseline, achieving an **R² of 0.9844**. This highlights the high degree of persistence in annual trade data and sets a formidable, non-trivial benchmark for any learning model.

2.  **Hybrid Features are Key:** While no learning model surpassed the Naive baseline in raw accuracy, the project successfully validated its core hypothesis: combining feature types improves performance. The final **XGBoost model, augmented with TGN-inspired embeddings, achieved an R² of 0.3197**, outperforming the same model with standard features (R² of 0.2835).

3.  **Learned Embeddings Add Value:** The GCN-LSTM model successfully generated dynamic node embeddings that captured complex, evolving network structures. Adding these learned features provided a **~3.6 percentage point increase in R²**, demonstrating their ability to capture predictive signals beyond what was available in the handcrafted features.

4.  **Deep Interpretability with SHAP:** Using SHAP, we confirmed that the model's predictions are driven by a mix of historical trade volume (`amount_rolling_mean`), dynamic network position (`importer_harmonic_centrality_...`), and the learned TGN embeddings. The analysis uncovered sophisticated interactions, such as the model learning that high trade volume has an even stronger positive impact when the importer is also a central player in the network.

## Project Pipeline

The project is structured as a sequence of eight Jupyter notebooks, designed to be run in order. Each notebook handles a distinct phase of the project, from data cleaning to final model interpretation.

### `1-preprocessing.ipynb`
*   **Data Acquisition:** Consolidates 37 yearly raw CSV files (1988-2024) from the `raw_import_data/` folder into a single DataFrame of over 795,000 records.
*   **Cleaning & Standardization:** Renames columns to a consistent schema (`importer`, `exporter`, `amount`, `year`) and filters out non-country entries.
*   **Feature Enrichment:** Adds geographic coordinates (latitude, longitude) for both importer and exporter countries.
*   **Significance Filtering:** Applies a "Top 20" filter to retain the most significant trade partners for each importer annually, reducing the dataset to 17,170 core observations.

### `2-feature_engineering.ipynb`
*   **Manual Time-Series Features:** Generates lagged trade amounts (up to 3 years) and 3-year rolling statistics (mean, std dev).
*   **Automated Time-Series Features:** Utilizes the `tsfresh` library to automatically extract **777 statistical features** for each of the 3,681 unique trade pairs, capturing a deep history of their relationship.
*   **Imputation:** Employs Multivariate Imputation by Chained Equations (MICE) to handle missing values introduced during feature engineering.

### `3-static_analysis.ipynb`
*   **Static Network Analysis:** Performs a deep-dive exploratory data analysis (EDA) on a 2023 network snapshot.
*   **Network Metrics:** Calculates graph properties like density (0.0747), clustering coefficient (0.2934), and centrality measures (PageRank, Betweenness, etc.).
*   **Community Detection:** Compares multiple algorithms (Louvain, Spectral Clustering) and identifies 4 distinct trade communities using the Greedy Modularity algorithm.
*   **Visualization:** Creates world map visualizations of the network structure, centrality scores, and detected communities.

### `4-dynamic_analysis.ipynb`
*   **Dynamic Network Construction:** Iteratively builds a trade graph for each year from 1988 to 2024.
*   **Dynamic Feature Engineering:** Calculates yearly centrality scores and community assignments for every country.
*   **Temporal Feature Creation:** Engineers rolling statistics on centrality scores and a `community_stability` feature to quantify how often a country changes trade blocs.
*   **Feature Lagging:** Lags all 34 new dynamic features by one year to prevent data leakage in forecasting models.

### `5-predictive_modeling.ipynb`
*   **Data Preparation:** Loads the fully-featured dataset, applies a `log1p` transformation to the skewed `amount` target variable, and selects 40 features suitable for forecasting.
*   **Time-Series Splitting:** Chronologically splits the data into training (until 2020), validation (2021-2022), and testing (2023-2024) sets.
*   **Baseline Implementation:** Develops and evaluates two key baselines: a Naive (Persistence) Forecast and a Historical Average Forecast.
*   **Data Persistence:** Saves the final data splits and the fitted `StandardScaler` for use in the next stage.

### `6-forecasting.ipynb`
*   **Advanced Model Training:** Implements, tunes, and evaluates three powerful tree-based ensemble models: LightGBM, Random Forest, and XGBoost.
*   **LSTM Implementation:** Develops and evaluates an LSTM network to serve as a deep learning baseline.
*   **Model Comparison:** Systematically compares all models against each other and the Part 5 baselines, identifying XGBoost as the top-performing learning model.

### `7-tgn_feature_exploration.ipynb`
*   **Graph Snapshot Preparation:** Transforms the dataset into a sequence of 37 yearly graph snapshots.
*   **Dynamic Embedding Learning:** Implements and trains a GCN-LSTM model to learn 32-dimensional node embeddings for each country-year, generating 6,660 embeddings in total.
*   **Feature Augmentation:** Lags the learned embeddings and merges them into the feature sets, increasing the feature count from 40 to 104.
*   **Performance Validation:** Retrains the best model (XGBoost) on the augmented data, demonstrating a significant performance improvement.

### `8-interpretation.ipynb`
*   **Qualitative Error Analysis:** Investigates *where* and *why* the best model makes mistakes by analyzing the largest prediction errors and their distribution over time and trade volume.
*   **XAI with SHAP:** Uses the SHAP library to explain the TGN-augmented XGBoost model's predictions.
*   **Global Interpretation:** Generates SHAP summary plots (bar, beeswarm) to identify globally important features and their effects.
*   **Local Interpretation:** Creates SHAP dependence and force plots to explain feature interactions and individual predictions, providing deep insight into the model's behavior.

## How to Run

### Prerequisites

This project was developed in Python 3.9. The following libraries are required. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn networkx python-louvain scikit-learn xgboost lightgbm tensorflow torch torch-geometric shap powerlaw
```
*Note: `torch` and `torch_geometric` may require specific installation steps depending on your CUDA version. Please refer to their official documentation.*

### Data
The raw data is expected to be a series of CSV files, one for each year (e.g., `1988.csv`, `1989.csv`, ...). These files must be placed in the `raw_import_data/` directory before running the first notebook.

### Execution Order
The Jupyter notebooks are designed to be run sequentially as they form a data processing and modeling pipeline. The output of each notebook is the input for the next.

1.  `1-preprocessing.ipynb`
2.  `2-feature_engineering.ipynb`
3.  `3-static_analysis.ipynb`
4.  `4-dynamic_analysis.ipynb`
5.  `5-predictive_modeling.ipynb`
6.  `6-forecasting.ipynb`
7.  `7-tgn_feature_exploration.ipynb`
8.  `8-interpretation.ipynb`

## Repository Structure
```
.
├── raw_import_data/
│   ├── 1988.csv
│   ├── 1989.csv
│   └── ... (and so on for all years)
├── 1-preprocessing.ipynb
├── 2-feature_engineering.ipynb
├── 3-static_analysis.ipynb
├── 4-dynamic_analysis.ipynb
├── 5-predictive_modeling.ipynb
├── 6-forecasting.ipynb
├── 7-tgn_feature_exploration.ipynb
├── 8-interpretation.ipynb
├── Report.pdf
├── MMDA Presentation.pdf
├── MMDA Presentation Notes.pdf
└── README.md
```
*Note: The pipeline will generate new directories (`processed_for_modeling/`, `trained_models/`, `tgn_embeddings/`) to store intermediate data, models, and embeddings.*
