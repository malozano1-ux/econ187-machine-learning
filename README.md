# ECON 187 — Machine Learning for Economists

Machine learning projects from ECON 187 at UCLA, applying classification and regression methods to financial markets, music streaming, and business analytics. Each project compares multiple models and evaluates performance using rigorous validation.

- **Course:** ECON 187 — Machine Learning for Economists, UCLA
- **Author:** Manuela Lozano
- **Tools:** R (`caret`, `randomForest`, `xgboost`, `gbm`, `glmnet`, `mgcv`, `MASS`, `pROC`, `nnet`, `e1071`)
- **Quarter:** Spring 2026

---

## Projects

### Project 1 — Predicting Stock Market Direction

**Data:** S&P 500 daily prices from Yahoo Finance via `quantmod` (2010–2024, ~3,522 observations)

**Research question:** To what extent can lagged financial indicators predict daily stock market direction, and do nonlinear classification methods provide meaningful improvements over linear models?

**Hypothesis:** Consistent with the weak-form Efficient Market Hypothesis — predictive performance will be limited and nonlinear models will not significantly outperform linear ones.

**Methods:**
- Binary outcome: "Up" or "Down" based on daily return
- Predictors: Lag1–Lag5 (previous 5 days' returns)
- Time-based train/test split (pre-2022 train, 2022+ test)
- Models: Logistic Regression, LDA, QDA, kNN (k tuned via 10-fold CV)
- ROC curves and AUC for threshold-independent evaluation

**Key results:**

| Model | Test Accuracy | CV AUC |
|---|---|---|
| Logistic Regression | ~52% | ~0.51 |
| LDA | ~52% | ~0.51 |
| QDA | ~50% | ~0.50 |
| kNN (CV-tuned) | ~53% | ~0.54 |

**Key findings:**
- All models perform close to random guessing — consistent with weak-form EMH
- Nonlinear models (QDA, kNN) do not meaningfully outperform linear ones
- Lagged returns provide minimal predictive signal for market direction

**Files:** `project1-stock-market/`

---

### Project 2 — Predicting Spotify Song Popularity

**Data:** 106,000+ Spotify songs with audio features (danceability, energy, loudness, acousticness, valence, tempo, genre, etc.)

**Research question:** Are nonlinear machine learning models more effective than traditional linear models at predicting Spotify song popularity using audio features?

**Methods:**
- 80/20 train/test split
- Linear models: Baseline OLS, Ridge, LASSO, Elastic Net, PCA regression
- Nonlinear models: Piecewise polynomial, Splines, GAM, Random Forest (n=100 trees on 20K sample), Bagging, XGBoost (200 rounds)
- Bootstrap RMSE confidence intervals (B=200) for model stability comparison

**Key results:**

| Model | Test RMSE | Test R² |
|---|---|---|
| Baseline OLS | highest | lowest |
| Ridge / LASSO / Elastic Net | similar to baseline | ~same |
| GAM | improved | improved |
| Bagging | better | better |
| XGBoost | strong | strong |
| **Random Forest** | **lowest** | **highest** |

**Key findings:**
- Random Forest achieves the best predictive performance — lowest RMSE, highest R²
- All regularized linear models perform nearly identically, suggesting the bottleneck is nonlinearity, not overfitting
- GAM smooth plots reveal nonlinear effects: popularity has threshold effects with danceability, loudness, and acousticness
- Top predictors (RF importance): acousticness, instrumentalness, energy, danceability, genre

**Files:** `project2-spotify/`

---

### Project 3 — Predicting Restaurant Success

**Data:** 1,000 restaurants with operational, marketing, financial, and customer-demand characteristics (kaggle-style synthetic dataset)

**Research question:** Can machine learning models accurately predict restaurant success using business, customer-demand, marketing, and operational characteristics?

**Methods:**
- Binary outcome: Successful vs. Unsuccessful (68%/32% imbalance)
- SMOTE applied to training data to address class imbalance
- 80/20 train/test split + 10-fold cross-validation (Neural Network, SVM)
- Models: Classification Tree, Random Forest (300 trees), Gradient Boosting (GBM, 300 rounds), Neural Network (`nnet`, size/decay tuned), SVM (radial kernel)
- Evaluation: accuracy, sensitivity, specificity, ROC-AUC, confusion matrices
- K-means clustering (k=4) for exploratory business profiling

**Key results:**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Classification Tree | ~96% | 0.978 |
| **Random Forest** | **~95.5%** | **~1.000** |
| Boosting (GBM) | ~94% | 0.997 |
| SVM (radial) | ~89% | 0.950 |
| Neural Network | ~87% | 0.945 |

**Key findings:**
- Random Forest and Boosting substantially outperform simpler methods
- Top predictors across all models: marketing budget, social media followers, weekend reservations, Google rating
- Digital visibility and customer engagement are the strongest predictors of restaurant success
- SMOTE effectively balanced training data without sacrificing test performance

**Files:** `project3-restaurant/`

---

## Repository Structure

```
econ187-machine-learning/
├── project1-stock-market/
│   ├── econ187_project1.Rmd       # Logistic, LDA, QDA, kNN + ROC analysis
│   └── econ187_project1.pdf       # Full written report
├── project2-spotify/
│   ├── econ187_project2.Rmd       # 11-model comparison + bootstrap CIs
│   ├── econ187_project2.pdf       # Full written report
│   └── spotify.csv                # Spotify audio features dataset
└── project3-restaurant/
    ├── econ187_project3.Rmd       # Tree/RF/Boosting/NN/SVM + SMOTE
    ├── econ187_project3.pdf       # Full written report
    └── Restaurant_Success_Dataset.csv
```

> **Note:** Project 1 pulls S&P 500 data live from Yahoo Finance via `quantmod::getSymbols()`. Projects 2 and 3 require the included CSV datasets.

> **Hardcoded paths:** Before running locally, update the `read.csv()` paths in `econ187_project2.Rmd` (line 48) and `econ187_project3.Rmd` (line 69) to point to your local file locations.
