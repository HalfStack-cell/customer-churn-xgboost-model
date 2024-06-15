 # Customer Churn Prediction using XGBoost

A machine learning project to predict customer churn using the Telco Customer Churn dataset from Kaggle. Utilizes the XGBoost algorithm for high-performance predictions. Includes data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Customer churn prediction is crucial for businesses to retain their customers and understand the reasons behind customer attrition. This project uses the Telco Customer Churn dataset to build a predictive model using the XGBoost algorithm. The model predicts whether a customer will churn based on various features like tenure, monthly charges, and contract type.

## Dataset
The dataset used in this project is the Telco Customer Churn dataset from Kaggle. It includes information about customer demographics, services subscribed, and account information.

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

## Installation
To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/HalfStack-cell/customer-churn-xgboost-model.git
    cd customer-churn-xgboost-model
    ```

2. **Set up the virtual environment**:
    ```bash
    conda create -n churn-env python=3.11
    conda activate churn-env
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Download and unzip the dataset**:
    Ensure the dataset zip file (`telco-customer-churn.zip`) is in the root directory. If not, download it from Kaggle and place it there.

2. **Run the model training script**:
    ```bash
    python customer-churn-xgboost-model.py
    ```

3. **Check the results**:
    The script will generate a `model_results.txt` file with the model's accuracy and classification report.

## Results
After running the script, you can find the results in `model_results.txt`. It includes:
- Best XGBoost Model Accuracy
- Precision, recall, and f1-score for each class
- Overall accuracy, macro average, and weighted average scores

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

