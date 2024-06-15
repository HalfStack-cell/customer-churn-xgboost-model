# Description: This script downloads the dataset from Kaggle, loads it into a Pandas DataFrame, preprocesses the data,
import os
import pandas as pd
import numpy as np
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek
import xgboost as xgb
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise
    
#Download the dataset from Kaggle
def download_dataset(config):
   
    try:
        if not os.path.exists(config['data']['dataset_zip']):
            logging.info("Downloading dataset...")
            subprocess.run(['kaggle', 'datasets', 'download', '-d', config['data']['kaggle_dataset']], check=True)
            logging.info("Dataset downloaded successfully!")
        else:
            logging.info("Dataset already downloaded!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading dataset: {e}")
        raise

def unzip_dataset(config):
    """Unzip the dataset if it isn't already unzipped."""
    try:
        if not os.path.exists(config['data']['dataset_csv']):
            logging.info("Unzipping dataset...")
            subprocess.run(['unzip', config['data']['dataset_zip']], check=True)
            logging.info("Dataset unzipped successfully!")
        else:
            logging.info("Dataset already extracted!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error unzipping dataset: {e}")
        raise

#Load the configuration file, load dataset and handle missing values
def load_dataset(config):
   
    try:
        logging.info("Loading the dataset...")
        df = pd.read_csv(config['data']['dataset_csv'])
        logging.info("Dataset loaded successfully!")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def handle_missing_values(df):
   
    try:
        logging.info("Handling missing values...")
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df = df.dropna(subset=['TotalCharges']).copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
        logging.info("Missing values handled.")
        return df
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
        raise

def encode_categorical_variables(df):

    try:
        logging.info("Encoding categorical variables...")
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
        df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
        df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
        df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
        df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines', 'OnlineSecurity',
                                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'],
                            drop_first=True)
        logging.info("Categorical variables encoded.")
        return df
    except Exception as e:
        logging.error(f"Error encoding categorical variables: {e}")
        raise

#Standardize the numerical features and split the data into training and testing sets
def standardize_numerical_features(df):
   
    try:
        logging.info("Standardizing numerical features...")
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        logging.info("Numerical features standardized.")
        return df
    except Exception as e:
        logging.error(f"Error standardizing numerical features: {e}")
        raise

def split_data(df):
   
    try:
        logging.info("Splitting the data into training and testing sets...")
        X = df.drop(columns=['customerID', 'Churn'])
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data split completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

#Evaluate the model, save the results, and check the results file
def evaluate_model(model, X_test, y_test):
   
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy}")
        logging.info("Classification Report:\n" + report)
        return accuracy, report
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_results(config, results):

    try:
        with open(config['output']['results_file'], 'w') as file:
            file.write(results)
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def check_results_file(config):
    results_file = config['output']['results_file']

    try:
        if os.path.exists(results_file):
            print(f"\nResults file '{results_file}' found. Displaying content:\n")
            with open(results_file, 'r') as file:
                print(file.read())
        else:
            print(f"\nResults file '{results_file}' not found.")
    except Exception as e:
        logging.error(f"Error checking results file: {e}")
        raise


#Main function to execute the end-to-end process
def main():

    try:
        config = load_config()
        
        download_dataset(config)
        unzip_dataset(config)
        
        df = load_dataset(config)
        df = handle_missing_values(df)
        df = encode_categorical_variables(df)
        df = standardize_numerical_features(df)
        
        X_train, X_test, y_train, y_test = split_data(df)
        
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        
        param_grid = config['model_params']['xgboost']
        xgb_model = xgb.XGBClassifier(random_state=42)
        random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, n_iter=50, random_state=42)
        random_search.fit(X_resampled, y_resampled)
        
        best_xgb_model = random_search.best_estimator_
        best_accuracy, best_report = evaluate_model(best_xgb_model, X_test, y_test)
        
        results = f"Best XGBoost Model Accuracy: {best_accuracy}\n{best_report}\n"
        
        save_results(config, results)
        logging.info("Process completed!")
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
    config = load_config()
    check_results_file(config)
