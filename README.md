# Diabetes Prediction Using Machine Learning

This project uses machine learning classification algorithms to predict whether a person is likely to have diabetes based on demographic, medical, lifestyle, and health-related features.

The project includes data preprocessing, BMI calculation, feature encoding, feature scaling, model training, and performance comparison using **Accuracy**, **ROC-AUC**, and **Training Time**.

## Project Overview

The dataset contains **100,000 records** with information including:

* Gender
* Age
* Hypertension
* Heart disease
* Smoking history
* HbA1c level
* Blood glucose level
* Height
* Weight
* Diabetes diagnosis

Height and weight are used to calculate **Body Mass Index (BMI)**, after which the original height and weight columns are removed from the modelling dataset.

## Machine Learning Models

The following classification algorithms are trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbours (KNN)
5. XGBoost

## Dataset

The project uses:

```text
diabetes_with_height_weight.csv
```

The dataset contains **100,000 rows** and includes height and weight measurements that are converted into BMI during preprocessing.

### Features

| Feature               | Description                                |
| --------------------- | ------------------------------------------ |
| `gender`              | Gender of the individual                   |
| `age`                 | Age of the individual                      |
| `hypertension`        | Whether the individual has hypertension    |
| `heart_disease`       | Whether the individual has heart disease   |
| `smoking_history`     | Smoking history category                   |
| `HbA1c_level`         | HbA1c measurement                          |
| `blood_glucose_level` | Blood glucose measurement                  |
| `height_m`            | Height in metres                           |
| `weight_kg`           | Weight in kilograms                        |
| `diabetes`            | Target variable indicating diabetes status |

## Data Preprocessing

### 1. Text Cleaning

The `gender` and `smoking_history` columns are cleaned by removing unnecessary whitespace.

### 2. Categorical Encoding

Categorical values are converted into numerical values.

Example:

```python
gender_map = {
    'Female': 0,
    'Male': 1,
    'Other': 2
}
```

Smoking history is also converted into numerical categories.

### 3. BMI Calculation

BMI is calculated using height and weight:

```python
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)
```

The BMI values are rounded to two decimal places.

After calculating BMI:

* `height_m` is removed
* `weight_kg` is removed
* `bmi` is used as the replacement feature

### 4. Feature Selection

The following features are used for model training:

```text
gender
age
hypertension
heart_disease
smoking_history
bmi
hba1c
glucose
```

The target variable is:

```text
diabetes
```

### 5. Feature Scaling

`StandardScaler` from Scikit-learn is used to standardise the input features.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 6. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

## Model Performance

The models are evaluated using Accuracy, ROC-AUC, and training time.

| Model               |   Accuracy |    ROC-AUC | Training Time |
| ------------------- | ---------: | ---------: | ------------: |
| **XGBoost**         | **97.14%** | **97.86%** |        5.65 s |
| Random Forest       |     97.07% |     96.36% |        9.06 s |
| KNN                 |     96.13% |     90.57% |        0.34 s |
| Logistic Regression |     95.87% |     96.12% |        0.20 s |
| Decision Tree       |     95.19% |     85.44% |        0.34 s |

> Performance values are based on the execution results recorded in the notebook and may vary slightly when the notebook is executed again.

## Results

Among the tested models, **XGBoost achieved the highest ROC-AUC and accuracy** in the recorded experiment.

### XGBoost

* Accuracy: **97.14%**
* ROC-AUC: **97.86%**
* Training Time: **5.65 seconds**

### Random Forest

* Accuracy: **97.07%**
* ROC-AUC: **96.36%**
* Training Time: **9.06 seconds**

XGBoost provided the best overall predictive performance in this experiment, while Logistic Regression had the shortest training time.

## Project Structure

```text
.
├── diabetes_prediction.ipynb
├── diabetes_with_height_weight.csv
├── .gitattributes
└── README.md
```

### `diabetes_prediction.ipynb`

Jupyter Notebook containing:

* Data loading
* Data exploration
* Data preprocessing
* BMI calculation
* Feature encoding
* Feature scaling
* Train-test splitting
* Machine learning model training
* Model evaluation
* Performance comparison
* Visualisations

### `diabetes_with_height_weight.csv`

Dataset containing the input features and diabetes target variable.

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Running the Project

Open the notebook using Jupyter:

```bash
jupyter notebook diabetes_prediction.ipynb
```

Alternatively, the notebook can be opened using **Google Colab**.

Make sure the following files are available in the same working directory:

```text
diabetes_prediction.ipynb
diabetes_with_height_weight.csv
```

Then run the notebook cells sequentially.

## Disclaimer

This project is intended for **educational and research purposes only**.

The predictions generated by the machine-learning models should not be considered medical diagnoses or a substitute for professional medical advice.

## Author

**Subin George**
