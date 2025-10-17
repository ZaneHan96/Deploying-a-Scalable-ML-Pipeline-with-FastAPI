# Model Card

For additional information, see the [Model Card paper by Mitchell et al. (2019)](https://arxiv.org/pdf/1810.03993.pdf).

---

## Model Details
- **Developer:** Educational project for model deployment using FastAPI  
- **Model type:** Logistic Regression  
- **Library:** Scikit-learn  
- **Task:** Binary classification — predict whether an individual's income exceeds \$50K per year  
- **Pipeline:** Uses `OneHotEncoder` for categorical features and `LabelBinarizer` for target labels  
- **Files:**  
  - `model.pkl` – trained model  
  - `encoder.pkl` – fitted encoder  
  - `label_binarizer.pkl` – fitted label binarizer  
- **Version:** 1.0  
- **Date:** October 2025  

---

## Intended Use
This model was built as part of an educational exercise on:
- Training, saving, and serving ML models  
- Building a FastAPI endpoint for inference  
- Evaluating model fairness and performance across data slices  

It is **not intended for production or any real-world decision-making**.

## Training Data
- **Source:** [UCI Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
- **Features:** Demographic and work-related attributes such as age, workclass, education, occupation, race, sex, and hours-per-week  
- **Target:** Binary label — income `<=50K` or `>50K`  
- **Split:** 80% training / 20% testing  
- **Processing:** Data encoded via `OneHotEncoder` and binarized labels  

---

## Evaluation Data
- **Type:** Stratified test split from the same dataset  
- **Purpose:** Evaluate model generalization and fairness across demographic slices  
- **Preprocessing:** Identical to training set encoding  

---


## Metrics
Performance on the overall test set:

- **Precision:** 0.55  
- **Recall:** 0.82  
- **F1-score:** 0.66  

Example performance on data slices (from `slice_output.txt`):

| Slice | Group | Precision | Recall | F1 |
|-------|--------|------------|---------|----|
| sex | Female | 0.53 | 0.70 | 0.60 |
| sex | Male | 0.55 | 0.83 | 0.66 |
| education | Masters | 0.73 | 0.94 | 0.82 |

For a complete list, refer to `slice_output.txt`.

---

## Ethical Considerations
- The training data reflects **historical social and economic biases**.  
- Model predictions may reproduce or amplify these biases.  
- **Protected attributes** such as race and sex are present in the dataset, and performance disparities exist across these groups.  
- The model should not be used to make real-world predictions or policy decisions.  

---

## Caveats and Recommendations
- Small or underrepresented groups produce unreliable metrics — interpret such results cautiously.  
- Retraining and auditing are recommended if adapting this model for further use.  
- Designed solely for learning purposes — not production deployment.  

