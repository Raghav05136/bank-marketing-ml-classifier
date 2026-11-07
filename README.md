# Bank Marketing: Term-Deposit Subscription Classifiers

## a. Problem Statement
A Portuguese bank runs phone-based direct marketing campaigns to sell term deposits. Calling every client is expensive, so the bank wants to predict, before or during a campaign, whether a client will subscribe (`y = yes`). This is a **binary classification** problem. Five classifiers are trained on the same data and compared, with Streamlit used to evaluate them interactively.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository, Bank Marketing dataset (S. Moro, R. Laureano, P. Cortez, 2011). Citation: *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*, ESM 2011.
- **File used:** `data/bank.csv` (random 10% sample of the full dataset, semicolon-separated)
- **Instances:** 4,521 (no duplicate rows found) (minimum required: 500)
- **Features:** 16 input features, 15 used after removing `duration` (minimum required: 12)
- **Target:** `y` (yes / no). Class balance: 4,000 no (88.5%) vs 521 yes (11.5%), so the data is **imbalanced**.
- **Feature types:**
  - Numeric: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`
  - Categorical: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **Dropped feature:** `duration` (last call duration in seconds). It is known only after a call ends, so using it would leak the outcome and make the model unusable for predicting before a call.
- **Missing values:** none in the source file. The pipeline still imputes (median for numeric, most-frequent for categorical) so unseen data does not crash it.
- **Preprocessing:** numeric features standardized; categorical features one-hot encoded; stratified 80/20 train/test split with `random_state=42` (3,616 train / 905 test).

## c. GitHub Repository Link
https://github.com/Raghav05136/bank-marketing-ml-classifier

## d. Models Used

### Evaluation metrics (held-out test set, 905 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8873 | 0.7376 | 0.5333 | 0.1538 | 0.2388 | 0.2429 |
| Decision Tree | 0.8796 | 0.6327 | 0.4359 | 0.1635 | 0.2378 | 0.2136 |
| KNN (k=7) | 0.8829 | 0.6450 | 0.4167 | 0.0481 | 0.0862 | 0.1097 |
| Naive Bayes | 0.8099 | 0.7166 | 0.2763 | 0.4038 | 0.3281 | 0.2274 |
| Support Vector Machine | 0.8884 | 0.6909 | 0.5600 | 0.1346 | 0.2171 | 0.2352 |

Precision, Recall and F1 are computed for the positive class (`yes`). Values are from `model/metrics_comparison.csv`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.738) and the best MCC (0.243) of the models. Its precision (0.53) is decent, but recall is low (0.15), so it misses most subscribers. It is a strong, interpretable baseline. |
| Decision Tree | Depth limited to 6 to reduce overfitting. Recall (0.16) and precision (0.44) are both modest, and AUC (0.63) is the lowest of all five, so its probability ranking is the weakest. Easy to explain but less reliable than the linear model. |
| KNN | Lowest recall (0.05) and F1 (0.09): it rarely predicts "yes", likely because the majority class dominates the neighbourhoods. Accuracy looks good (0.88) but this is misleading on imbalanced data. Distance-based, so it also depends on feature scaling. |
| Naive Bayes | Highest recall (0.40) and F1 (0.33) of the models, so it catches more subscribers, at the cost of lower precision (0.28) and accuracy (0.81). Its independence assumption is unlikely to hold exactly for this data, yet it ranks clients reasonably well (AUC 0.72). Useful if the bank prefers to call more people. |
| Support Vector Machine | Highest accuracy (0.888) and precision (0.56), but recall (0.13) is low, so it is conservative and misses many subscribers. Its AUC (0.69) is reasonable. Probability outputs (needed for AUC) make it more costly to train than the other models, which is manageable at this dataset size. |
| **Overall winner** | **Logistic Regression.** It has the best MCC (0.243) and AUC (0.738), a balanced precision and recall, and it is fast and interpretable. Accuracy is misleading here (predicting "no" always gives 88.5%), so MCC and AUC were used to decide. If the bank's priority is maximizing the number of subscribers it reaches, Naive Bayes is the better choice because it has the highest recall and F1. |

## Project Structure
```
APP.py                  Streamlit app (upload CSV, pick model, metrics, confusion matrix, comparison tab)
Train_Models.py         Trains and evaluates all 5 models, saves them to model/
requirements.txt        Python dependencies
Bank Dataset/bank.csv           Source dataset (UCI, semicolon-separated)
test_data.csv           Held-out test split used in the app (features + y)
model/                  Saved pipelines (*.joblib), label encoder, metrics_comparison.csv
```

## How to Run
```bash
pip install -r requirements.txt
python train_models.py        # trains models, writes model/ and test_data.csv
streamlit run app.py
```

**Live app:** https://bank-marketing-ml-classifier-4pniebagm2shtlvevxcv2j.streamlit.app/
