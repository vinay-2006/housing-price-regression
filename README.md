# Housing Price Regression

**End-to-end regression system for predicting housing prices with leakage-free pipelines and production-ready code structure.**

This project demonstrates professional ML engineering practices: modular preprocessing, reusable pipelines, systematic evaluation, and disciplined experimentation workflows.

---

## 📋 Project Overview

This project implements a **baseline-first, correctness-focused** regression system. The objective is not model sophistication, but establishing:
- ✅ **Proper ML workflow discipline** (train/test splits, no data leakage)
- ✅ **Modular, testable components** (preprocessing, pipelines, evaluation)
- ✅ **Baseline validation** (compare against trivial predictors)
- ✅ **Interpretable diagnostics** (residual analysis, outlier detection)

All code prioritizes **interpretability, correctness, and reviewability** over complexity.

---

## 🗂️ Repository Structure

```
housing-price-regression/
│
├─ data/
│   ├─ raw/                          # Original, unmodified datasets
│   └─ processed/                    # Cleaned and transformed data
│
├─ src/
│   ├─ preprocessing.py              # Feature engineering and transformers
│   ├─ pipeline.py                   # End-to-end modeling pipelines
│   └─ evaluation.py                 # Metrics, baselines, residual analysis
│
├─ notebooks/
│   └─ housing_price_regression.ipynb  # Execution and analysis layer
│
├─ README.md                         # Project documentation
└─ requirements.txt                  # Python dependencies
```

### Design Principles
- **No business logic in notebooks**: All preprocessing, modeling, and evaluation code lives in `src/`
- **Leakage-free pipelines**: All transformations are fit only on training data
- **Separation of concerns**: Preprocessing, modeling, and evaluation are independent modules
- **Reproducible workflows**: Fixed seeds, documented transforms, transparent data flows

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/vinay-2006/housing-price-regression.git
cd housing-price-regression
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare data:**
   - Place raw housing dataset in `data/raw/`
   - Processed data will be saved to `data/processed/` during execution

4. **Run the notebook:**
```bash
jupyter notebook notebooks/housing_price_regression.ipynb
```

---

## 📦 Module Documentation

### `src/preprocessing.py`
**Purpose:** Feature engineering and data transformation

**Key Functions:**
- `build_preprocessor()`: Constructs a `ColumnTransformer` for numerical and categorical features
  - Numerical: median imputation → standard scaling
  - Categorical: most-frequent imputation → one-hot encoding
- `split_features_target()`: Separates features (X) and target (y) with missing value handling

**Guarantees:**
- No data leakage (transformers fit only on training data)
- Type-safe column handling
- Explicit missing value strategy

---

### `src/pipeline.py`
**Purpose:** End-to-end modeling pipelines

**Key Functions:**
- `build_regression_pipeline()`: Combines preprocessing and linear regression into a single pipeline

**Benefits:**
- One-line training: `pipeline.fit(X_train, y_train)`
- Consistent preprocessing across train/test
- Easy to extend (swap `LinearRegression` for other models)

---

### `src/evaluation.py`
**Purpose:** Model diagnostics and baseline comparison

**Key Functions:**
- `evaluate_regression()`: Computes MAE, RMSE, and R² metrics
- `baseline_predictors()`: Generates trivial baselines (mean/median prediction)
- `plot_residuals()`: Visualizes residual patterns and distributions
- `detect_outliers()`: Identifies extreme residuals using z-score thresholding

**Use Case:**
Always compare your model against baseline predictors. If your model doesn't beat the mean baseline, something is wrong.

---

## 📓 Notebook Workflow

The `housing_price_regression.ipynb` notebook follows this structure:

1. **Setup**: Import modules from `src/`, configure paths
2. **Data Loading**: Load raw data without preprocessing
3. **Feature-Target Split**: Separate X and y using `split_features_target()`
4. **Train-Test Split**: Use sklearn's `train_test_split()` for validation
5. **Pipeline Building**: Construct leakage-free pipeline with `build_regression_pipeline()`
6. **Training**: Fit pipeline on training data only
7. **Evaluation**: 
   - Compute metrics on train and test sets
   - Compare against baseline predictors
   - Plot residual diagnostics
8. **Analysis**: Identify failure modes, outliers, and next steps

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| **Core ML** | scikit-learn |
| **Numerical Computing** | NumPy |
| **Data Manipulation** | Pandas |
| **Visualization** | Matplotlib |
| **Environment** | Jupyter Notebook |
| **Code Structure** | Modular Python (src/) |

---

## 📊 Key Features

### 1. Leakage-Free Preprocessing
```python
# All transformations are fit ONLY on training data
pipeline = build_regression_pipeline(numeric_features, categorical_features)
pipeline.fit(X_train, y_train)  # Learns scaling/encoding from training set
y_pred = pipeline.predict(X_test)  # Applies learned transforms to test set
```

### 2. Baseline Validation
```python
# Always compare against trivial predictors
baseline_mean = baseline_predictors(y_train, strategy="mean")
baseline_pred = [baseline_mean] * len(y_test)
baseline_metrics = evaluate_regression(y_test, baseline_pred)
```

### 3. Residual Diagnostics
```python
# Identify systematic errors
plot_residuals(y_test, y_pred)
outlier_indices = detect_outliers(residuals, threshold=3.0)
```

---

## 📈 Evaluation Strategy

### Metrics
- **MAE (Mean Absolute Error)**: Average prediction error in original units
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily
- **R² (Coefficient of Determination)**: Variance explained by the model

### Baseline Comparison
Every model is compared against:
- **Mean baseline**: Predict the training mean for all test samples
- **Median baseline**: Predict the training median for all test samples

**Rule:** If your model doesn't significantly beat the baseline, investigate before adding complexity.

### Residual Analysis
- **Residuals vs Predictions**: Should show no systematic patterns (constant variance)
- **Residual Distribution**: Should be approximately normal
- **Outlier Detection**: Identify samples with extreme errors for further investigation

---

## 🔮 Future Work

Potential extensions are intentionally deferred.

This project serves as a **baseline and diagnostic anchor**. More complex models are justified only if residual analysis demonstrates clear limitations of linear assumptions.

---

## 📂 Data

### Expected Format
- **Location**: `data/raw/`
- **Format**: CSV file with housing features and target price column
- **Required columns**: Defined in notebook (numeric and categorical features)

### Example Schema
```
# Numeric features: square_footage, bedrooms, bathrooms, year_built, etc.
# Categorical features: neighborhood, property_type, etc.
# Target: price
```

---

## 📬 Contact

**Author:** Vinay Reddy  
**GitHub:** [@vinay-2006](https://github.com/vinay-2006)  
**LinkedIn:** [vinay-boppidi-b5216b351](https://www.linkedin.com/in/vinay-boppidi-b5216b351/)  
**Email:** boppidivinayred@gmail.com

---

## 📝 Notes

### Why Linear Regression?
This project uses linear regression as the **baseline model**. The focus is on:
- Establishing correct workflow patterns
- Understanding residual behavior
- Building interpretable systems

More complex models are only justified after exhausting baseline analysis.

### Why Modular Code?
- **Testability**: Each function can be unit tested independently
- **Reusability**: Preprocessing and evaluation logic work with any sklearn model
- **Reviewability**: Clear function signatures with docstrings
- **Maintainability**: Changes to preprocessing don't affect evaluation logic

---

## 📄 License

This project is open-source and available for educational purposes.
