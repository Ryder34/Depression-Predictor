#!/usr/bin/env python
# coding: utf-8

# In[6]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from bayes_opt import BayesianOptimization

import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,Dataset
from pytorch_tabnet.tab_model import TabNetClassifier
from tab_transformer_pytorch import TabTransformer
import warnings
warnings.filterwarnings("ignore")


# ## Preprocessing

# ### Creating a custom cleaner which can be added to Pipelines
# #### All steps which involve removal of rows must be done before pipelining

# In[7]:


class CustomCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer that:
      - Replaces '?' with NaN.
      - Keeps only rows with Profession == 'Student'.
      - Drops columns: 'id', 'City', and 'Profession'.
      - One-hot encodes 'Gender' (dropping the first dummy).
      - Ordinal-encodes:
          * 'Dietary Habits' with order: Healthy, Moderate, Unhealthy, Others.
          * 'Have you ever had suicidal thoughts ?' with order: No, Yes.
          * 'Sleep Duration' with order: 'More than 8 hours', '7-8 hours', '5-6 hours', 'Less than 5 hours', Others.
      - Performs mean encoding on 'Degree' using the target (Depression) mean.
      - Converts all columns to numeric where applicable.
      - Fills remaining NaNs using the median calculated during fit. Handles cases where median might be NaN.
    """
    def __init__(self, target_column='Depression'):
        self.target_column = target_column
        self.fill_values_ = None
        self.feature_columns_ = None  
        self.scaler_ = StandardScaler()


    def _clean_dataframe(self, df):
        if 'Gender' in df.columns:
            df = pd.get_dummies(df, columns=['Gender'], drop_first=True, prefix='Gender')
        mapping_dicts = {
            # 'Gender':{"Male":0,"Female":1},
            'Dietary Habits': {"Healthy": 0, "Moderate": 1, "Unhealthy": 2, "Others": 3},
            'Have you ever had suicidal thoughts ?': {"No": 0, "Yes": 1},
            'Sleep Duration': {"More than 8 hours": 0, "7-8 hours": 1, "5-6 hours": 2, "Less than 5 hours": 3, "Others": 4},
        }
        for col, mapping in mapping_dicts.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        if 'Degree' in df.columns and 'Depression' in df.columns:
            degree_target_mean = df.groupby('Degree')['Depression'].mean()
            df['Degree'] = df['Degree'].map(degree_target_mean)
        if 'City' in df.columns and 'Depression' in df.columns:
            degree_target_mean = df.groupby('City')['Depression'].mean()
            df['City'] = df['City'].map(degree_target_mean)
            
        df = df.apply(pd.to_numeric, errors='coerce')
        return df


    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_column] = y.copy()
        
        df = self._clean_dataframe(df)
        
        y_cleaned = None
        if self.target_column in df.columns:
            y_cleaned = df[self.target_column].values
            df.drop(columns=[self.target_column], inplace=True)
        
        self.fill_values_ = df.median().fillna(0)
        df.fillna(self.fill_values_, inplace=True)
        
        self.feature_columns_ = df.columns
        
        self.scaler_.fit(df)
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_column] = y.copy()
        
        df = self._clean_dataframe(df)
        
        y_cleaned = None
        if self.target_column in df.columns:
            y_cleaned = df[self.target_column].values
            df.drop(columns=[self.target_column], inplace=True)
        
        df.fillna(self.fill_values_, inplace=True)
        X_scaled = self.scaler_.transform(df)
        
        return X_scaled
    def get_feature_names_out(self, input_features=None):
        return self.feature_columns_


# #### Creating Pipeline templates

# In[8]:


def get_pipeline_preprocessor():
    """Allows more elemnts to be added to pipeline if necessary"""
    return Pipeline([
        ('cleaner', CustomCleaner()),
    ])
def get_poly_preprocessor(X_fit, exclude_cols_from_poly, degree):
    """
    Creates a preprocessor that cleans, applies PolynomialFeatures (excluding specified columns),
    and then scales everything. Imputation is handled within the cleaner.
    Args:
        X_fit (pd.DataFrame): Data used to fit the cleaner (to get column names).
        exclude_cols_from_poly (list): List of column names *after cleaning* to exclude from poly expansion.
        degree (int): Degree for PolynomialFeatures.
    """
    temp_cleaner = CustomCleaner()
    try:
        temp_cleaner.fit(X_fit.copy()) 
        cleaned_cols = list(temp_cleaner.columns_)
        if not cleaned_cols:
             print("Warning: Cleaner resulted in no columns during poly preprocessor setup.")
             # Return a simple pipeline if no columns remain
             return Pipeline([('cleaner', CustomCleaner()), ('scaler', StandardScaler())])
    except Exception as e:
        print(f"Error fitting temporary cleaner in get_poly_preprocessor: {e}")
        return Pipeline([('cleaner', CustomCleaner()), ('scaler', StandardScaler())])


    poly_cols = [col for col in cleaned_cols if col not in exclude_cols_from_poly]
    passthrough_cols = [col for col in cleaned_cols if col in exclude_cols_from_poly]

    transformers = []
    if poly_cols:
        transformers.append(('poly', PolynomialFeatures(degree=degree, include_bias=False), poly_cols))
    if passthrough_cols:
        transformers.append(('pass', 'passthrough', passthrough_cols))

    if not transformers:
         print("Warning: No columns identified for polynomial features or passthrough.")
         # Fallback to basic cleaner + scaler
         return Pipeline([('cleaner', CustomCleaner()), ('scaler', StandardScaler())])

    poly_transformer = ColumnTransformer(transformers, remainder='passthrough')


    pipeline = Pipeline([
        ('cleaner', CustomCleaner()), 
        ('poly_features', poly_transformer), 
        ('scaler', StandardScaler())
    ])
    return pipeline
def get_pca_preprocessor(n_components):
    """
    Creates a pipeline that:
        - Cleans the data using CustomCleaner
        - Scales the features using StandardScaler
        - Applies PCA for dimensionality reduction

    Args:
        n_components (int or float): Number of PCA components to keep.
                                     Can be int (number of features) or float (percentage variance to retain).
    Returns:
        sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('cleaner', CustomCleaner()),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42))
    ])
    return pipeline


# #### Loading and performing initial preprocessing (Especially which involves row removal)

# In[9]:


df_raw = pd.read_csv("Team01_Dataset.csv")

target_column = 'Depression'
if target_column not in df_raw.columns:
    print(f"Error: Target column '{target_column}' not found in the CSV.")
    
df_raw.replace('?', np.nan, inplace=True)
        
# Filter for students and drop unnecessary columns
if 'Profession' in df_raw.columns:
    df_raw = df_raw[df_raw['Profession'] == 'Student'].copy()
    df_raw.drop(columns=['id', 'Profession'], inplace=True, errors='ignore')
else:
    df_raw.drop(columns=['id'], inplace=True, errors='ignore')

if target_column in df_raw.columns:
    df_raw.dropna(subset=[target_column], inplace=True)
X = df_raw.drop(columns=[target_column]) 
y = df_raw[target_column]                

# Splitting data before applying preprocessing pipelines in tuning functions
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
        # Handling potential stratify error 
        print(f"Error during train_test_split (possibly due to stratify): {e}")
        print("Trying split without stratify...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e_nostrat:
            print(f"Error during non-stratified split: {e_nostrat}. Cannot proceed.")
            

preprocessor_factory = get_pipeline_preprocessor()

results = {}


# ## Code for Shapely Analysis

# In[10]:


def plot_shap_values(pipeline, X, explainer_type='auto', sample_size=100):
    """
    Parameters:
      - pipeline: A scikit-learn Pipeline that contains  preprocessor and final model.
      - X: The input features (as a DataFrame or array) to compute SHAP values on.
      - explainer_type: Type of explainer to use ('auto', 'kernel', or 'tree').
          * 'auto': Automatically selects KernelExplainer for non-tree models.
          * 'tree': Uses TreeExplainer (suitable for tree-based models).
          * 'kernel': Uses KernelExplainer.
      - sample_size: Number of samples to use for explanation (KernelExplainer can be slow on large datasets).

    Returns:
      None

    Usage:
        plot_shap_values(trained_pipeline, X_test)
    """
    # If your pipeline contains a preprocessor, extract and apply it
    preprocessor = None
    for name, step in pipeline.steps:
        if hasattr(step, "transform") and name != pipeline.steps[-1][0]:
            preprocessor = step
        else:
            # Assume the final step is the model
            model = step

    if preprocessor is not None:
        # Transform the data using the preprocessor part of the pipeline.
        X_processed = preprocessor.transform(X)
        if isinstance(X_processed, np.ndarray) and hasattr(preprocessor, "feature_columns_"):
            X_processed = pd.DataFrame(X_processed, columns=preprocessor.feature_columns_)
    else:
        # No preprocessor found, use X directly.
        X_processed = X.copy()

    if preprocessor is not None:
        X_processed = preprocessor.transform(X)

    # Try to extract meaningful feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    # Create DataFrame with feature names for SHAP
    print(feature_names)
    X_processed = pd.DataFrame(X_processed, columns=X.columns)

    # Take random sample
    X_sample = X_processed.sample(n=min(sample_size, len(X_processed)), random_state=42)

    # Initialize the appropriate SHAP explainer based on explainer_type or model attributes.
    if explainer_type == 'auto':
        explainer = shap.KernelExplainer(model.predict, X_sample)
    elif explainer_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif explainer_type == 'kernel':
        explainer = shap.KernelExplainer(model.predict, X_sample)
    else:
        raise ValueError("Unknown explainer type specified. Use 'auto', 'tree' or 'kernel'.")

    shap_values = explainer.shap_values(X_sample)
    print(X.columns)

    if isinstance(shap_values, list) and len(shap_values) > 0:
        shap.summary_plot(shap_values[0], X_sample)
        return shap_values[0]
    else:
        shap.summary_plot(shap_values, X_sample)
        return shap_values


# # Testing

# ### Logistic Regression

# In[11]:


def tune_logistic_regression(X_train, y_train, preprocessor):
    # Cross Validation function
    def lr_cv(C):
        
        current_prep = get_pipeline_preprocessor() 
        pipe = Pipeline([
            ('prep', current_prep), 
            ('clf', LogisticRegression(C=C, solver='liblinear', random_state=42, max_iter=1000))
        ])
        try:
             score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
             return score if not np.isnan(score) else 0.0
        except ValueError as ve:
             return 0.0
        except Exception as e:
             return 0.0

    # Bayesian Optimization over C
    optimizer = BayesianOptimization(lr_cv, {'C': (0.001, 10)}, random_state=42, verbose=0)
    try:
        optimizer.maximize(init_points=5, n_iter=10)
    except Exception as e:
        print(f"BayesianOptimization failed for Logistic Regression: {e}")
        return None, {}, 0.0 
    if not optimizer.max:
         print("Logistic Regression optimization found no maximum.")
         return None, {}, 0.0

    best_params = optimizer.max['params']
    best_params['solver'] = 'liblinear'
    best_params['random_state'] = 42
    best_params['max_iter'] = 1000

    # Create the final pipeline with best params
    final_preprocessor = get_pipeline_preprocessor()
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', LogisticRegression(**best_params))
    ])
    try:
        best_pipe.fit(X_train, y_train)
        plot_shap_values(best_pipe, X_train, explainer_type='auto', sample_size=50)
        return best_pipe, best_params, optimizer.max['target']
    except Exception as e:
        print(f"Error fitting final Logistic Regression pipeline: {e}")
        return None, best_params, optimizer.max['target']


# #### Hyperparameter Tuning

# In[12]:


# Logistic Regression
print("Tuning Logistic Regression...")

lr_model, lr_params, lr_score = tune_logistic_regression(X_train.copy(), y_train.copy(), preprocessor_factory)
results['Logistic Regression'] = {'model': lr_model, 'params': lr_params, 'cv_f1': lr_score}
print(f"  Done. Best F1: {lr_score:.4f}, Params: {lr_params}")


# #### Hyperparamter Tuning

# ### SVM (rbf)

# In[13]:


def tune_svm(X_train, y_train, preprocessor):
    def svm_cv(C, gamma):
        current_prep = get_pipeline_preprocessor() 
        pipe = Pipeline([
            ('prep', current_prep),
            ('clf', SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42))
        ])
        try:
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
            return score if not np.isnan(score) else 0.0
        except ValueError as ve:
            return 0.0
        except Exception as e:
            return 0.0
    optimizer = BayesianOptimization(svm_cv, {'C': (0.1, 10), 'gamma': (0.0001, 1)},random_state=42, verbose=0)
    # try:
    #     optimizer.maximize(init_points=5, n_iter=10)
    # except Exception as e:
    #     print(f"BayesianOptimization failed for SVM: {e}")
    #     return None, {}, 0.0

    # if not optimizer.max:
    #     print("SVM optimization found no maximum.")
    #     return None, {}, 0.0

    # best_params = optimizer.max['params']
    best_params={}
    final_preprocessor = get_pipeline_preprocessor() 
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    try:
        best_pipe.fit(X_train, y_train)
        return best_pipe, best_params, f1_score(best_pipe.predict(X_train),y_train)
    except Exception as e:
        print(f"Error fitting final SVM pipeline: {e}")
        return None, best_params, optimizer.max['target']


# #### Hyperparamter Tuning

# In[14]:


# Too Slow so run at ur risk
# # SVM (non-linear)
print("\nTuning SVM...")
svm_model, svm_params, svm_score = tune_svm(X_train.copy(), y_train.copy(), preprocessor_factory)
results['SVM'] = {'model': svm_model, 'params': svm_params, 'cv_f1': svm_score}
print(f"  Done. Best F1: {svm_score:.4f}, Params: {svm_params}")


# ### Decision Tree

# In[15]:


def tune_decision_tree(X_train, y_train, preprocessor):
    def dt_cv(max_depth, min_samples_split):
        int_max_depth = int(round(max_depth))
        int_min_samples_split = int(round(min_samples_split))
        if int_max_depth < 1: int_max_depth = 1
        if int_min_samples_split < 2: int_min_samples_split = 2

        current_prep = get_pipeline_preprocessor()
        pipe = Pipeline([
            ('prep', current_prep),
            ('clf', DecisionTreeClassifier(max_depth=int_max_depth,min_samples_split=int_min_samples_split,random_state=42))
        ])
        try:
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
            return score if not np.isnan(score) else 0.0
        except ValueError as ve:
            return 0.0
        except Exception as e:
            return 0.0
    # Search Space
    optimizer = BayesianOptimization(dt_cv, {'max_depth': (3, 15), 'min_samples_split': (2, 20)},random_state=42, verbose=0)
    try:
        optimizer.maximize(init_points=5, n_iter=10)
    except Exception as e:
        print(f"BayesianOptimization failed for Decision Tree: {e}")
        return None, {}, 0.0

    if not optimizer.max:
        print("Decision Tree optimization found no maximum.")
        return None, {}, 0.0

    best_params = optimizer.max['params']
    best_params['max_depth'] = int(round(best_params['max_depth']))
    best_params['min_samples_split'] = int(round(best_params['min_samples_split']))
    if best_params['max_depth'] < 1: best_params['max_depth'] = 1
    if best_params['min_samples_split'] < 2: best_params['min_samples_split'] = 2

    final_preprocessor = get_pipeline_preprocessor()
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', DecisionTreeClassifier(**best_params, random_state=42))
    ])
    try:
        best_pipe.fit(X_train, y_train)
        plot_shap_values(best_pipe, X_train, explainer_type='auto', sample_size=50)

        return best_pipe, best_params, optimizer.max['target']
    except Exception as e:
        print(f"Error fitting final Decision Tree pipeline: {e}")
        return None, best_params, optimizer.max['target']


# #### Hyperparameter Tuning

# In[16]:


# Decision Tree
print("\nTuning Decision Tree...")
dt_model, dt_params, dt_score = tune_decision_tree(X_train.copy(), y_train.copy(), preprocessor_factory)
results['Decision Tree'] = {'model': dt_model, 'params': dt_params, 'cv_f1': dt_score}
print(f"  Done. Best F1: {dt_score:.4f}, Params: {dt_params}")


# ### Random Forest

# In[17]:


def tune_random_forest(X_train, y_train, preprocessor):
    def rf_cv(n_estimators, max_depth):
        int_n_estimators = int(round(n_estimators))
        int_max_depth = int(round(max_depth))
        if int_n_estimators < 10: int_n_estimators = 10
        if int_max_depth < 1: int_max_depth = 1

        current_prep = get_pipeline_preprocessor() 
        pipe = Pipeline([
            ('prep', current_prep),
            ('clf', RandomForestClassifier(n_estimators=int_n_estimators,max_depth=int_max_depth,random_state=42, n_jobs=-1))])
        try:
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1', n_jobs=-1).mean()
            return score if not np.isnan(score) else 0.0
        except ValueError as ve:
            return 0.0
        except Exception as e:
             return 0.0

    # Search Space
    optimizer = BayesianOptimization(rf_cv, {'n_estimators': (50, 300), 'max_depth': (3, 15)},random_state=42, verbose=0)
    try:
        optimizer.maximize(init_points=5, n_iter=10)
    except Exception as e:
        print(f"BayesianOptimization failed for Random Forest: {e}")
        return None, {}, 0.0

    if not optimizer.max:
        print("Random Forest optimization found no maximum.")
        return None, {}, 0.0

    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(round(best_params['n_estimators']))
    best_params['max_depth'] = int(round(best_params['max_depth']))
    if best_params['n_estimators'] < 10: best_params['n_estimators'] = 10
    if best_params['max_depth'] < 1: best_params['max_depth'] = 1


    final_preprocessor = get_pipeline_preprocessor() 
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', RandomForestClassifier(**best_params, random_state=42, n_jobs=-1))
    ])
    try:
        best_pipe.fit(X_train, y_train)
        plot_shap_values(best_pipe, X_train, explainer_type='auto', sample_size=50)
        return best_pipe, best_params, optimizer.max['target']
    except Exception as e:
        print(f"Error fitting final Random Forest pipeline: {e}")
        return None, best_params, optimizer.max['target']


# #### Hyperparamter Tuning

# In[18]:


# Random Forest
print("\nTuning Random Forest...")
rf_model, rf_params, rf_score = tune_random_forest(X_train.copy(), y_train.copy(), preprocessor_factory)
results['Random Forest'] = {'model': rf_model, 'params': rf_params, 'cv_f1': rf_score}
print(f"  Done. Best F1: {rf_score:.4f}, Params: {rf_params}")


# ### XGBoost

# In[58]:


def tune_xgboost(X_train, y_train, preprocessor):
    def xgb_cv(learning_rate, max_depth, n_estimators, subsample, colsample_bytree):
        int_max_depth = int(round(max_depth))
        int_n_estimators = int(round(n_estimators))
        if int_max_depth < 1: int_max_depth = 1
        if int_n_estimators < 10: int_n_estimators = 10

        params = {'learning_rate': learning_rate,
                  'max_depth': int_max_depth,
                  'n_estimators': int_n_estimators,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                  'random_state': 42,
                  'use_label_encoder': False,
                  'eval_metric': 'logloss'}

        current_prep = get_pipeline_preprocessor() 
        pipe = Pipeline([
            ('prep', current_prep),
            ('clf', XGBClassifier(**params))
        ])
        y_train_int = y_train.astype(int)
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(pipe, X_train, y_train_int, cv=cv, scoring='f1').mean()
            return score if not np.isnan(score) else 0.0
        except ValueError as ve:
            return 0.0
        except Exception as e:
             return 0.0

    pbounds = {'learning_rate': (0.01, 0.3),
               'max_depth': (3, 10),
               'n_estimators': (50, 800),
               'subsample': (0.5, 1.0),
               'colsample_bytree': (0.5, 1.0)}
    optimizer = BayesianOptimization(xgb_cv, pbounds, random_state=42, verbose=0)

    try:
        optimizer.maximize(init_points=5, n_iter=10)
    except Exception as e:
        print(f"BayesianOptimization failed for XGBoost: {e}")
        return None, {}, 0.0

    if not optimizer.max:
         print("XGBoost optimization found no maximum.")
         return None, {}, 0.0

    best_params = optimizer.max['params']
    best_params['max_depth'] = int(round(best_params['max_depth']))
    best_params['n_estimators'] = int(round(best_params['n_estimators']))
    if best_params['max_depth'] < 1: best_params['max_depth'] = 1
    if best_params['n_estimators'] < 10: best_params['n_estimators'] = 10

    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'

    final_preprocessor = get_pipeline_preprocessor() 
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', XGBClassifier(**best_params))
    ])
    try:
        best_pipe.fit(X_train, y_train.astype(int))
        shap_vals=plot_shap_values(best_pipe, X_train, explainer_type='tree', sample_size=50)
        return best_pipe, best_params, optimizer.max['target'],shap_vals
    except Exception as e:
        print(f"Error fitting final XGBoost pipeline: {e}")
        return None, best_params, optimizer.max['target'],shap_vals


# #### Hyperparamter Tuning

# In[59]:


# XGBoost
print("\nTuning XGBoost...")
xgb_model, xgb_params, xgb_score ,xgb_shap_vals= tune_xgboost(X_train.copy(), y_train.copy(), preprocessor_factory)
results['XGBoost'] = {'model': xgb_model, 'params': xgb_params, 'cv_f1': xgb_score}
print(f"  Done. Best F1: {xgb_score:.4f}, Params: {xgb_params}")


# In[62]:


xgb_shap_vals[0]


# In[21]:


def tune_tabnet_classifier(X_train, y_train, preprocessor):
    """
    Tune TabNetClassifier using Bayesian Optimization.
    The hyperparameters tuned include:
      - learning_rate: learning rate for TabNet, passed via optimizer_params.
      - gamma: sparsity regularization weight.
      - n_steps: number of steps in the feature transformer (converted to int).
    Uses cross-validation F1-score as the optimization metric.
    """
    # Function that computes cross-validated F1 score for given hyperparameters.
    def tabnet_cv(learning_rate, gamma, n_steps):
        current_prep = get_pipeline_preprocessor()
        # Build parameter dictionary. Ensure n_steps is an integer.
        params = {
            "optimizer_params": {"lr": learning_rate},  # Updated: learning rate goes here.
            "gamma": gamma,
            "n_steps": int(round(n_steps)),
            "verbose": 0,
            "seed": 42,
        }
        # Create the pipeline with preprocessor and TabNetClassifier.
        pipe = Pipeline([
            ('prep', current_prep),
            ('clf', TabNetClassifier(**params))
        ])
        try:
            # Use cross_val_score to evaluate the TabNet model.
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
            return score if not np.isnan(score) else 0.0
        except Exception as e:
            return 0.0

    # Bayesian Optimization over TabNet hyperparameters.
    optimizer = BayesianOptimization(
        f=tabnet_cv, 
        pbounds={
            'learning_rate': (0.001, 0.1),
            'gamma': (1.0, 1.5),
            'n_steps': (3, 10)  # n_steps is an integer but we search in a continuous range.
        },
        random_state=42, 
        verbose=0
    )
    try:
        optimizer.maximize(init_points=5, n_iter=1)
    except Exception as e:
        print(f"BayesianOptimization failed for TabNet: {e}")
        return None, {}, 0.0 
    if not optimizer.max:
        print("TabNet optimization found no maximum.")
        return None, {}, 0.0

    best_params = optimizer.max['params']
    best_params['n_steps'] = int(round(best_params['n_steps']))
    best_params['verbose'] = 0
    best_params['seed'] = 42
    # Place the learning_rate inside optimizer_params.
    best_params['optimizer_params'] = {"lr": best_params.pop('learning_rate')}

    # Build the final pipeline with the optimized hyperparameters.
    final_preprocessor = get_pipeline_preprocessor()
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', TabNetClassifier(**best_params))
    ])
    try:
        # Note: TabNet expects numpy arrays for fitting.
        best_pipe.fit(X_train, y_train)
        # Evaluate on training set.
        preds = best_pipe.predict(X_train)
        train_f1 = f1_score(y_train, preds)
        train_accuracy=accuracy_score(y_train,preds)

        return best_pipe, best_params, train_f1, train_accuracy
    except Exception as e:
        print(f"Error fitting final TabNet pipeline: {e}")
        return None, best_params, 0.0, 0.0
def train_tabnet_classifier(X_train, y_train, preprocessor):
    """
    Train a TabNetClassifier without Bayesian optimization or cross-validation.
    Uses fixed hyperparameters and returns:
      - The final pipeline,
      - The hyperparameters used (in the same structure as the Bayesian tuned version), and
      - The training F1 score.
    """
    # Define default fixed hyperparameters.
    default_params = {
        "optimizer_params": {"lr": 0.01},  # Learning rate placed inside optimizer_params
        "gamma": 1.3,                    # Sparsity regularization parameter
        "n_steps": 5,                    # Number of feature transformer steps
        "verbose": 0,
        "seed": 42,
    }
    
    # Build the preprocessing pipeline using the given factory function.
    final_preprocessor = get_pipeline_preprocessor()
    
    # Build the final pipeline combining preprocessor and TabNetClassifier.
    pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', TabNetClassifier(**default_params))
    ])
    
    try:
        # TabNetClassifier expects numpy arrays.
        pipe.fit(X_train, y_train)
        plot_shap_values(pipe, X_train, explainer_type='auto', sample_size=50)
        preds = pipe.predict(X_train)
        train_f1 = f1_score(y_train, preds)
        train_accuracy=accuracy_score(y_train,preds)
    except Exception as e:
        print(f"Error during TabNet training: {e}")
        return None, default_params, 0.0, 0.0

    return pipe, default_params, train_f1, train_accuracy


# In[22]:


print("Tuning TabNetClassifier...")
# tabnet_model, tabnet_params, tabnet_score = tune_tabnet_classifier(X_train.copy(), y_train.copy(), get_pipeline_preprocessor())
tabnet_model, tabnet_params, tabnet_score ,tabnet_accuracy= train_tabnet_classifier(X_train.copy(), y_train.copy(), get_pipeline_preprocessor())
results['TabNetClassifier'] = {'model': tabnet_model, 'params': tabnet_params, 'cv_f1': tabnet_score}

print(f"  Done. Best Training F1: {tabnet_score:.4f},Best Training Accuracy: {tabnet_accuracy:.4f} Params: {tabnet_params}")




# In[23]:


from sklearn.base import BaseEstimator
from flaml import AutoML

class FLAMLEstimator(BaseEstimator):
    def __init__(self, task="classification", time_budget=60, metric="f1", eval_method="cv"):
        self.task = task
        self.time_budget = time_budget
        self.metric = metric
        self.eval_method = eval_method
        self.automl = AutoML()




    def fit(self, X, y):
        self.automl.fit(
            X, y,
            task=self.task,
            time_budget=self.time_budget,
            metric=self.metric,
            eval_method=self.eval_method,
            estimator_list=["lgbm", "rf", "extra_tree"]
        )
        return self


    def predict(self, X):
        return self.automl.predict(X)


# In[24]:


pipeline = Pipeline([
    ('prep', get_pipeline_preprocessor()),
    ('automl', FLAMLEstimator())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_train)
train_f1 = f1_score(y_train, y_pred)
print(f"  Done. Best Training F1: {train_f1:.4f}")

y_pred = pipeline.predict(X_test)
test_f1 = f1_score(y_test ,y_pred)
print(f"  Done. Best Test F1: {test_f1:.4f}") 


# In[25]:


plot_shap_values(pipeline, X_train, explainer_type='auto', sample_size=200)
flaml_step = pipeline.named_steps['automl']
automl = flaml_step.automl  # Access AutoML instance
results['AutoML'] = {'model': pipeline, 'params': automl.best_config, 'cv_f1': train_f1}

print("Best estimator:", automl.best_estimator)
print("Best config:", automl.best_config)
print("Best model:", automl.model)
# Best config: {'n_estimators': 127, 'num_leaves': 4, 'min_child_samples': 10, 'learning_rate': 0.19920783759255895, 'log_max_bin': 6, 'colsample_bytree': 0.8967318034082727, 'reg_alpha': 0.020453547647346384, 'reg_lambda': 0.0018763588487861376}


# In[26]:


pre=pipeline.named_steps['prep'].named_steps['cleaner']
pre.get_feature_names_out()


# In[27]:


plt.barh(pre.get_feature_names_out(), automl.feature_importances_)


# ## Clustering Methods

# In[28]:


from collections import Counter

def purity_score(y_true, y_pred):
    clusters = {}
    for pred, true in zip(y_pred, y_true):
        if pred not in clusters:
            clusters[pred] = []
        clusters[pred].append(true)
    
    total = 0
    for members in clusters.values():
        most_common = Counter(members).most_common(1)[0][1]
        total += most_common
    return total / len(y_true)


# ## DBSCAN

# In[29]:


from sklearn.cluster import DBSCAN

def tune_density_clustering(X_train,y_train):
    # Cross-validation function for DBSCAN
    def dbscan_cv(eps, min_samples):
        eps = max(eps, 0.01)
        min_samples = int(round(min_samples))

        current_prep = CustomCleaner()  # Use your existing preprocessor
        try:
            X_processed = current_prep.fit_transform(X_train)

            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_processed)

            # Valid cluster check
            if len(set(labels)) <= 1 or all(label == -1 for label in labels):
                return 0.0

            score = purity_score(y_train, labels)
            return score if not np.isnan(score) else 0.0
        except Exception as e:
            print(f"Error during DBSCAN CV: {e}")
            return 0.0

    # Bayesian Optimization over eps and min_samples
    optimizer = BayesianOptimization(
        dbscan_cv,
        {'eps': (0.1, 3.0), 'min_samples': (3, 15)},
        random_state=42,
        verbose=0
    )

    try:
        optimizer.maximize(init_points=5, n_iter=10)
    except Exception as e:
        print(f"Bayesian Optimization failed for DBSCAN: {e}")
        return None, {}, 0.0

    if not optimizer.max:
        print("DBSCAN optimization found no maximum.")
        return None, {}, 0.0

    best_params = optimizer.max['params']
    best_params['eps'] = float(best_params['eps'])
    best_params['min_samples'] = int(round(best_params['min_samples']))

    # Create final pipeline
    final_preprocessor = CustomCleaner()
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']))
    ])

    try:
        X_processed = final_preprocessor.fit_transform(X_train)
        labels = best_pipe.named_steps['clf'].fit_predict(X_processed)
        print(f"Best DBSCAN score: {optimizer.max['target']:.4f}, Params: {best_params}")
        return best_pipe, best_params, optimizer.max['target']
    except Exception as e:
        print(f"Error fitting final DBSCAN pipeline: {e}")
        return None, best_params, optimizer.max['target']


# In[30]:


print("DBscan tunning in progress\n")
best_dbscan_pipe, dbscan_params, dbscan_score = tune_density_clustering(X_train,y_train)
results['Clusstering_density'] = {'model': best_dbscan_pipe, 'params': dbscan_params, 'cv_f1': dbscan_score}
print(f"  Done. Best F1: {dbscan_score:.4f}, Params: {dbscan_params}")


# ## KMeans

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import numpy as np

def tune_kmeans_clustering(X_train):
    def kmeans_cv(n_clusters):
        n_clusters = int(n_clusters)
        if n_clusters <= 1:
            return 0.0
        try:
            # Pipeline with cleaning, scaling and clustering
            pipe = Pipeline([
                ('prep', CustomCleaner()),
                ('scaler', StandardScaler()),
                ('cluster', KMeans(n_clusters=n_clusters, random_state=42))
            ])

            # Fit and transform
            X_processed = pipe.named_steps['prep'].fit_transform(X_train)
            X_scaled = pipe.named_steps['scaler'].fit_transform(X_processed)

            labels = pipe.named_steps['cluster'].fit_predict(X_scaled)
            score = purity_score(y_train, labels)
            return score
        except Exception as e:
            return 0.0

    optimizer = BayesianOptimization(f=kmeans_cv, pbounds={'n_clusters': (2, 10)}, random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=10)

    if not optimizer.max:
        print("KMeans optimization found no maximum.")
        return None, {}, 0.0

    best_n_clusters = int(optimizer.max['params']['n_clusters'])

    best_pipeline = Pipeline([
        ('prep', CustomCleaner()),
        ('scaler', StandardScaler()),
        ('cluster', KMeans(n_clusters=best_n_clusters, random_state=42))
    ])

    try:
        best_pipeline.fit(X_train)
    except Exception as e:
        print(f"Error fitting final KMeans pipeline: {e}")
        return None, {'n_clusters': best_n_clusters}, optimizer.max['target']

    return best_pipeline, {'n_clusters': best_n_clusters}, optimizer.max['target']


# In[32]:


print("Kmeans tunnning in progress\n")
kmeans_model, kmeans_params, kmeans_score = tune_kmeans_clustering(X_train.copy())
results['Clusstering_mean'] = {'model': kmeans_model, 'params': kmeans_params, 'cv_f1': kmeans_score}
print(f"  Done. Best F1: {kmeans_score:.4f}, Params: {kmeans_params}")


# ## KNN Model

# In[33]:


from sklearn.neighbors import KNeighborsClassifier

def tune_knn(X_train, y_train, X_test, preprocessor):
    # Categorical options
    weight_options = ['uniform', 'distance']
    metric_options = ['euclidean', 'manhattan']

    weight_map = {i: w for i, w in enumerate(weight_options)}
    metric_map = {i: m for i, m in enumerate(metric_options)}

    def knn_cv(n_neighbors, weights_idx, metric_idx):
        n_neighbors = int(n_neighbors)
        weights = weight_map[int(round(weights_idx))]
        metric = metric_map[int(round(metric_idx))]

        current_prep = get_pipeline_preprocessor()
        pipe = Pipeline([
            ('prep', current_prep),
            ('clf', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric))
        ])

        try:
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
            return score if not np.isnan(score) else 0.0
        except Exception:
            return 0.0

    optimizer = BayesianOptimization(
        knn_cv,
        {
            'n_neighbors': (1, 30),
            'weights_idx': (0, len(weight_options) - 1),
            'metric_idx': (0, len(metric_options) - 1)
        },
        random_state=42,
        verbose=0
    )

    try:
        optimizer.maximize(init_points=5, n_iter=10)
    except Exception as e:
        print(f"BayesianOptimization failed for KNN: {e}")
        return None, {}, 0.0

    if not optimizer.max:
        print("KNN optimization found no maximum.")
        return None, {}, 0.0

    best_raw = optimizer.max['params']
    best_params = {
        'n_neighbors': int(best_raw['n_neighbors']),
        'weights': weight_map[int(round(best_raw['weights_idx']))],
        'metric': metric_map[int(round(best_raw['metric_idx']))]
    }

    # best params
    final_preprocessor = get_pipeline_preprocessor()
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('clf', KNeighborsClassifier(**best_params))
    ])

    try:
        best_pipe.fit(X_train, y_train)
        # plot_shap_values(best_pipe, X_train, explainer_type='auto', sample_size=50)
        return best_pipe, best_params, optimizer.max['target']
    except Exception as e:
        print(f"Error fitting final KNN pipeline: {e}")
        return None, best_params, optimizer.max['target']


# In[34]:


knn_model, knn_params, knn_score = tune_knn(X_train.copy(),y_train.copy(), X_test.copy(), preprocessor_factory)
results['Clusstering_mean'] = {'model': knn_model, 'params': knn_params, 'cv_f1': knn_score}
print(f"  Done. Best F1: {knn_score:.4f}, Params: {knn_params}")


# ### MLP

# In[35]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np

def train_ann_sklearn(X_train, y_train, ann_preprocessor, hidden_dim1=64, hidden_dim2=64, lr=0.01, max_iter=200):
    try:
        X_train_transformed = ann_preprocessor.transform(X_train.copy())
    except Exception as e:
        return None, {}, 0.0

    if np.isnan(X_train_transformed).any():
        return None, {}, 0.0

    # Define 2-layer MLP with hidden layers
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_dim1, hidden_dim2),
        learning_rate_init=lr,
        max_iter=max_iter,
        random_state=42,
        solver='adam',
        early_stopping=True
    )
    final_preprocessor= get_pipeline_preprocessor()
    best_pipe = Pipeline([
        ('prep', final_preprocessor),
        ('mlp', model)
    ])
    best_pipe.fit(X_train, y_train)
    preds = best_pipe.predict(X_train)
    train_f1 = f1_score(y_train, preds, zero_division=0)

    return best_pipe, {
        'hidden_dim1': hidden_dim1,
        'hidden_dim2': hidden_dim2,
        'lr': lr,
        'max_iter': max_iter
    }, train_f1


# In[36]:


print("\nTraining ANN with sklearn MLPClassifier...")
ann_preprocessor_fitted = get_pipeline_preprocessor()
ann_preprocessor_fitted.fit(X_train.copy(), y_train.copy())

dimensions_1 = [64, 128]
dimensions_2 = [32, 64, 128]
learning_rates = [0.002, 0.01, 0.05]

ann_models = {}
ann_parameters = {}
ann_f1 = []

best_f1 = -1
best_model = None
best_params = {}
best_preprocessor = None

try:
    for d1 in dimensions_1:
        for d2 in dimensions_2:
            for lr in learning_rates:
                ann_model, ann_params, ann_train_f1 = train_ann_sklearn(
                    X_train.copy(), y_train.copy(),
                    ann_preprocessor_fitted,
                    hidden_dim1=d1, hidden_dim2=d2, lr=lr,
                    max_iter=300
                )
                print(f"  Finished training with dims=({d1},{d2}), lr={lr} => F1={ann_train_f1:.4f}")

                ann_models[((d1, d2), lr)] = ann_model
                ann_parameters[((d1, d2), lr)] = ann_params
                ann_f1.append(ann_train_f1)

                if ann_train_f1 > best_f1:
                    best_f1 = ann_train_f1
                    best_model = ann_model
                    best_params = ann_params
                    best_preprocessor = ann_preprocessor_fitted
    best_params = {'hidden_dim1': 128, 'hidden_dim2': 32, 'lr': 0.02}
    print(f"  Done. Training F1s: {ann_f1}\n Best F1: {best_f1}\n Best Params: {best_params}\n")

except Exception as e:
    print(f"  Skipping ANN: Error during training: {e}\n")
    best_model, best_params, best_f1, best_preprocessor = None, {}, 0.0, None
plot_shap_values(best_model, X_train, explainer_type='auto', sample_size=50)
results['ANN'] = {
    'model': best_model,
    'params': best_params,
    'cv_f1': best_f1,
    'preprocessor': best_preprocessor
}


# ## Ensemble Methods

# ### Voting Classifier 'soft'

# In[37]:


results['AutoML']


# In[38]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#Individual models

log_reg = LogisticRegression(C= np.float64(3.7460266483547775), solver= 'liblinear',random_state= 42, max_iter= 1000)
ann = MLPClassifier(hidden_layer_sizes=(128,32), learning_rate_init=0.02)
xgb = XGBClassifier(colsample_bytree= np.float64(0.8384682993637083), learning_rate= np.float64(0.08516215128043891), max_depth= 3, n_estimators= 325, subsample= np.float64(0.6652566894203296), random_state= 42, use_label_encoder= False, eval_metric= 'logloss')
rdfr = RandomForestClassifier(max_depth=10,n_estimators=191, random_state=42)
dctr = DecisionTreeClassifier(max_depth= 7, min_samples_split= 19,random_state=42)
knn = KNeighborsClassifier(n_neighbors=29, weights='uniform', metric='manhattan')
tbnt = TabNetClassifier(gamma= 1.3, n_steps= 5, verbose= 0, seed= 42)
lgbm= LGBMClassifier(n_estimators=127,num_leaves=4,min_child_samples=10,learning_rate=np.float64(0.19920783759255895),log_max_bin=6,colsample_bytree=np.float64(0.8967318034082727),reg_alpha=np.float64(0.020453547647346384),reg_lambda=np.float64(0.0018763588487861376))
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('ann', ann),
        ('xgb', xgb),
        ('rdfr', rdfr),
        # ('dctr', dctr),
        ('lgbm', lgbm)
    ],
    voting='soft'  # 'hard' / 'soft'
)
final_preprocessor=get_pipeline_preprocessor()
test_pipe = Pipeline([
        ('prep',final_preprocessor),
        ('clf', voting_clf)
    ])
# Fit the ensemble
test_pipe.fit(X_train, y_train)

# Predict
y_pred = test_pipe.predict(X_train)
ensemble_f1 = f1_score(y_train, y_pred)
plot_shap_values(test_pipe, X_train, explainer_type='auto', sample_size=50)
results['Ensemble'] = {'model': test_pipe, 'cv_f1': ensemble_f1}
print(f"F1-score:{ensemble_f1}")


# ## Stacking

# In[54]:


from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def tune_stacking_model(X_train, y_train, preprocessor):
    # Base estimator (first layer): Logistic Regression
    log_reg = LogisticRegression(C= np.float64(3.7460266483547775), solver= 'liblinear',random_state= 42, max_iter= 1000)
    ann = MLPClassifier(hidden_layer_sizes=(128,32), learning_rate_init=0.02)
    xgb = XGBClassifier(colsample_bytree= np.float64(0.8384682993637083), learning_rate= np.float64(0.08516215128043891), max_depth= 3, n_estimators= 325, subsample= np.float64(0.6652566894203296), random_state= 42, use_label_encoder= False, eval_metric= 'logloss')
    rdfr = RandomForestClassifier(max_depth=10,n_estimators=191, random_state=42)
    dctr = DecisionTreeClassifier(max_depth= 7, min_samples_split= 19,random_state=42)
    knn = KNeighborsClassifier(n_neighbors=29, weights='uniform', metric='manhattan')
    tbnt = TabNetClassifier(gamma= 1.3, n_steps= 5, verbose= 0, seed= 42)
    lgbm= LGBMClassifier(n_estimators=127,num_leaves=4,min_child_samples=10,learning_rate=np.float64(0.19920783759255895),log_max_bin=6,colsample_bytree=np.float64(0.8967318034082727),reg_alpha=np.float64(0.020453547647346384),reg_lambda=np.float64(0.0018763588487861376))
    # Final estimator (second layer): Random Forest
    final_estimator  = LGBMClassifier(n_estimators=127,num_leaves=4,min_child_samples=10,learning_rate=np.float64(0.19920783759255895),log_max_bin=6,colsample_bytree=np.float64(0.8967318034082727),reg_alpha=np.float64(0.020453547647346384),reg_lambda=np.float64(0.0018763588487861376))


    # Stacking classifier
    stack_clf = StackingClassifier(
        estimators=[('logreg', log_reg),('ann',ann),('xgb',xgb),('lgbm',lgbm),('rdfr',rdfr)],
        final_estimator=final_estimator,
        passthrough=False,  # Set True if you want raw features + predictions
        cv=5
    )

    # Full pipeline with preprocessor
    full_pipe = Pipeline([
        ('prep', preprocessor),
        ('stack', stack_clf)
    ])

    try:
        # f1_score = cross_val_score(full_pipe, X_train, y_train, cv=5, scoring='f1').mean()
        full_pipe.fit(X_train, y_train)
        shap_vals=plot_shap_values(full_pipe, X_train, explainer_type='auto', sample_size=50)

        f1_scores=f1_score(full_pipe.predict(X_train),y_train)
        return full_pipe, f1_scores,shap_vals
    except Exception as e:
        print(f"Error during stacking model training: {e}")
        return None, 0.0, 0


# In[55]:


# Stacking
print("Training Stacking Model (LogReg + RF)...")

stack_model, stack_f1, shap_vals = tune_stacking_model(X_train.copy(), y_train.copy(), preprocessor_factory)
results['Stacking (LR+RF)'] = {
    'model': stack_model,
    'cv_f1': stack_f1
}

print(f"  Done. Best F1: {stack_f1:.4f}")


# In[41]:


# f1_score(stack_model.predict(X_test),y_test) #with mlp as master 


# In[42]:


# f1_score(stack_model.predict(X_test),y_test) #with logreg as master 


# In[43]:


# f1_score(stack_model.predict(X_test),y_test) #with xgb as master 


# In[44]:


# f1_score(stack_model.predict(X_test),y_test) #with lgbm as master 


# ## Evaluation

# In[45]:


def evaluate_model(model, X_test, y_test, preprocessor=None, model_type='sklearn'):
    if model is None:
        print("Evaluation skipped: Model is None.")
        return None

    try:
        if model_type == 'sklearn':
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
            else:
                try:
                    y_scores = model.decision_function(X_test)
                    y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                except AttributeError:
                    y_prob = None 
        elif model_type == 'pytorch':
            if preprocessor is None:
                raise ValueError("Preprocessor must be provided for PyTorch model evaluation.")

            X_test_trans = preprocessor.transform(X_test.copy()) 
            if np.isnan(X_test_trans).any():
                 print("Error: NaNs detected in PyTorch test data after preprocessing!")
            X_tensor = torch.tensor(X_test_trans.astype(np.float32))
            model.eval() 
            with torch.no_grad():
                outputs = model(X_tensor)
                y_pred = outputs.round().numpy().flatten()
                y_prob = outputs.numpy().flatten() 
        else:
            raise ValueError("Unknown model_type specified.")

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
            
        }
        return metrics

    except Exception as e:
        print(f"Error during model evaluation ({model_type}): {e}")
        return None 


def plot_confusion(cm, title):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


# ### Summary generation

# In[46]:


summary = []
valid_results = {k: v for k, v in results.items() if v and v.get('cv_f1') is not None} # Filtering out failed results

for key, val in valid_results.items():
    summary.append({
        'Model': key,
        'CV_F1': round(val['cv_f1'], 4),
        'Parameters': val.get('params', {})
    })

if not summary:
    print("\nNo models were successfully trained or tuned. Cannot generate summary.")
    exit()

summary_df = pd.DataFrame(summary).sort_values(by='CV_F1', ascending=False).reset_index(drop=True)

print("\n--- Cross-Validation / Training Metrics Summary ---")
print(summary_df)

# Plot comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='CV_F1', y='Model', data=summary_df, orient='h')
plt.title("CV F1 Score / Training F1 (ANN) Comparison")
plt.xlabel("F1 Score")
plt.ylabel("Model")
plt.xlim(0, max(1.0, summary_df['CV_F1'].max() * 1.1)) 
for index, row in summary_df.iterrows():
    plt.text(row['CV_F1'] + 0.01, index, f"{row['CV_F1']:.3f}", color='black', va='center')
plt.tight_layout()
plt.show()


# ### Evaluation on best model

# In[47]:


if summary_df.empty or summary_df.iloc[0]['CV_F1'] == 0.0:
        print("\nNo model performed adequately based on CV/Training F1. Cannot select best model.")
        exit()

best_model_name = summary_df.iloc[0]['Model']
print(f"\n--- Evaluating Best Model ({best_model_name}) on Test Set ---")

test_metrics = None
model_info = results.get(best_model_name)

if not model_info or model_info.get('model') is None:
        print(f"Best model ({best_model_name}) object is missing or None. Cannot evaluate.")
elif best_model_name == 'ANNs':
    best_model_ann = model_info['model']
    best_preprocessor_ann = model_info.get('preprocessor') 
    if best_model_ann and best_preprocessor_ann:
        test_metrics = evaluate_model(best_model_ann, X_test.copy(), y_test.copy(),
                                        preprocessor=best_preprocessor_ann, model_type='pytorch')
    else:
            print("ANN model or its specific preprocessor was not available for evaluation.")
else:
    best_model_sklearn = model_info['model']
    test_metrics = evaluate_model(best_model_sklearn, X_test.copy(), y_test.copy(), model_type='sklearn')


# Display test results
if test_metrics:
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        if metric != 'Confusion Matrix':
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}:\n{value}")

    # Plot confusion matrix for the best model
    plot_confusion(test_metrics['Confusion Matrix'], f"Test Set: {best_model_name} Confusion Matrix")
else:
    print(f"\nCould not evaluate the best model ({best_model_name}) on the test set.")


# In[48]:


import matplotlib.pyplot as plt

print("\n--- Evaluating All Models on Test Set ---\n")
all_test_metrics = {}

for model_name, model_info in results.items():
    print(f"\nEvaluating Model: {model_name}")
    
    model = model_info.get('model', None)
    preprocessor = model_info.get('preprocessor', None)

    if model is None:
        print(f"  Skipping {model_name} – model object is missing or None.")
        continue

    try:
        if model_name == 'ANNs':
            if preprocessor is None:
                print("  ANN model's preprocessor is missing. Skipping.")
                continue
            metrics = evaluate_model(model, X_test.copy(), y_test.copy(), 
                                     preprocessor=preprocessor, model_type='pytorch')
        else:
            metrics = evaluate_model(model, X_test.copy(), y_test.copy(), model_type='sklearn')

        if metrics is not None and isinstance(metrics, dict):
            all_test_metrics[model_name] = metrics
            print("  Test Set Metrics:")
            for metric, value in metrics.items():
                if metric != 'Confusion Matrix':
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}:\n{value}")
            plot_confusion(metrics['Confusion Matrix'], f"Test Set: {model_name} Confusion Matrix")
        else:
            print(f"  Evaluation returned None or invalid metrics for model: {model_name}")

    except Exception as e:
        print(f"  Error while evaluating {model_name}: {e}")


# In[49]:


all_test_metrics


# In[50]:


f1_scores = {model: metrics['F1'] 
             for model, metrics in all_test_metrics.items() 
             if isinstance(metrics, dict) and 'F1' in metrics}

if f1_scores:
    models = list(f1_scores.keys())
    scores = list(f1_scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, scores, color='mediumslateblue', edgecolor='black')
    plt.ylabel('Model')
    plt.xlabel('F1 Score')
    plt.title('F1 Score on Test Set for All Models')
    plt.xlim(0, 1)

    # Annotate each bar with the F1 score
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("\nNo F1 scores available to plot.")


# In[51]:


f1_scores = {model: metrics['Accuracy'] 
             for model, metrics in all_test_metrics.items() 
             if isinstance(metrics, dict) and 'Accuracy' in metrics}

if f1_scores:
    models = list(f1_scores.keys())
    scores = list(f1_scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, scores, color='mediumslateblue', edgecolor='black')
    plt.ylabel('Model')
    plt.xlabel('Accuracy Score')
    plt.title('Accuracy Score on Test Set for All Models')
    plt.xlim(0, 1)

    # Annotate each bar with the F1 score
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("\nNo F1 scores available to plot.")


# In[52]:


f1_scores = {model: metrics['Precision'] 
             for model, metrics in all_test_metrics.items() 
             if isinstance(metrics, dict) and 'Precision' in metrics}

if f1_scores:
    models = list(f1_scores.keys())
    scores = list(f1_scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, scores, color='mediumslateblue', edgecolor='black')
    plt.ylabel('Model')
    plt.xlabel('Precision Score')
    plt.title('Precision Score on Test Set for All Models')
    plt.xlim(0, 1)

    # Annotate each bar with the F1 score
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("\nNo F1 scores available to plot.")


# In[53]:


f1_scores = {model: metrics['Recall'] 
             for model, metrics in all_test_metrics.items() 
             if isinstance(metrics, dict) and 'Recall' in metrics}

if f1_scores:
    models = list(f1_scores.keys())
    scores = list(f1_scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, scores, color='mediumslateblue', edgecolor='black')
    plt.ylabel('Model')
    plt.xlabel('Recall Score')
    plt.title('Recall Score on Test Set for All Models')
    plt.xlim(0, 1)

    # Annotate each bar with the F1 score
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("\nNo F1 scores available to plot.")


# In[57]:


shap_vals[0]


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from semopy import Model
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Load data
df_raw = pd.read_csv("Team01_Dataset.csv")
target_column = 'Depression'

# Verify target column exists
if target_column not in df_raw.columns:
    raise ValueError(f"Error: Target column '{target_column}' not found in the CSV.")

# Replace '?' with NaN
df_raw.replace('?', np.nan, inplace=True)

# Filter for students and drop unnecessary columns
df_raw = df_raw[df_raw['Profession'] == 'Student'].copy()
df_raw.drop(columns=['id', 'Profession'], inplace=True, errors='ignore')

# Drop rows where target is NaN
df_raw.dropna(subset=[target_column], inplace=True)

# Check sample size
print(f"Sample size: {len(df_raw)}")
if len(df_raw) < 100:
    print("Warning: Sample size is very small for SEM. Expect unstable results.")
elif len(df_raw) < 200:
    print("Warning: Sample size may be insufficient for reliable SEM.")

# Define custom cleaner (modified to avoid mean-encoding)
class CustomCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Depression'):
        self.target_column = target_column
        self.fill_values_ = None
        self.feature_columns_ = None
        self.scaler_ = StandardScaler()

    def _clean_dataframe(self, df):
        # One-hot encode Gender
        if 'Gender' in df.columns:
            df = pd.get_dummies(df, columns=['Gender'], drop_first=True, prefix='Gender')

        # Ordinal encoding
        mappings = {
            'Dietary Habits': {"Healthy": 0, "Moderate": 1, "Unhealthy": 2, "Others": 3},
            'Have you ever had suicidal thoughts ?': {"No": 0, "Yes": 1},
            'Sleep Duration': {"More than 8 hours": 0, "7-8 hours": 1, "5-6 hours": 2, "Less than 5 hours": 3, "Others": 4},
            'Family History of Mental Illness': {"No": 0, "Yes": 1}  # Assuming Yes/No
        }
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Drop City and Degree to avoid high dimensionality
        if 'City' in df.columns:
            df = df.drop(columns=['City'])
        if 'Degree' in df.columns:
            df = df.drop(columns=['Degree'])

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_column] = y
        df = self._clean_dataframe(df)
        if self.target_column in df.columns:
            df.drop(columns=[self.target_column], inplace=True)
        self.fill_values_ = df.median().fillna(0)
        df.fillna(self.fill_values_, inplace=True)
        self.feature_columns_ = df.columns
        self.scaler_.fit(df)
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_column] = y
        df = self._clean_dataframe(df)
        if self.target_column in df.columns:
            df.drop(columns=[self.target_column], inplace=True)
        df.fillna(self.fill_values_, inplace=True)
        X_scaled = self.scaler_.transform(df[self.feature_columns_])
        return X_scaled

# Apply preprocessing
X = df_raw.drop(columns=[target_column])
y= df_raw[target_column]
cleaner = CustomCleaner(target_column='Depression')
cleaner.fit(X, y)
X_scaled = cleaner.transform(X)
feature_columns = cleaner.feature_columns_
df_processed = pd.DataFrame(X_scaled, columns=feature_columns)
df_processed[target_column] = y.values

# Diagnose data issues
print("\nData Summary:")
print(df_processed.describe())

# Check for zero or near-zero variance
variances = df_processed.var()
print("\nVariances:")
print(variances)
if (variances < 1e-6).any():
    print("Warning: Some variables have near-zero variance, which may cause numerical issues.")

# Check correlations
print("\nCorrelation Matrix:")
corr_matrix = df_processed.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[ ]:


# Rename columns for SEM compatibility
df_sem = df_processed.rename(columns={
    'Have you ever had suicidal thoughts ?': 'Have_you_ever_had_suicidal_thoughts_',
    'Family History of Mental Illness': 'Family_History_of_Mental_Illness',
    'Sleep Duration': 'Sleep_Duration',
    'Academic Pressure': 'Academic_Pressure',
    'Financial Stress': 'Financial_Stress',
    'Study Satisfaction': 'Study_Satisfaction',
    'Work/Study Hours': 'Work_Study_Hours',
    "Dietary Habits":"Dietary_Habits",
    "Job Satisfaction":"Job_Satisfaction",
    "Work Pressure":"Work_Pressure"
    
})
# df_sem = df_sem.loc[:, df_sem.var() > 1e-6]
# df_sem = df_sem.drop(columns=['Financial_Stress'])
# Ultra-simplified SEM model
model_desc = """
# Structural model
Depression ~ Academic_Pressure
Depression ~ Have_you_ever_had_suicidal_thoughts_
Depression ~ Work_Study_Hours
Stress =~ Academic_Pressure + Financial_Stress 
"""

# Fit SEM model with increased regularization
model = Model(model_desc)
try:
    model.fit(df_sem)  # Increased smoothing
    results = model.inspect()
    print("\nSEM Results:")
    print(results)
    
except np.linalg.LinAlgError as e:
    print(f"Error: Covariance matrix issue persists: {e}")
    print("Suggestions:")
    print("- Check sample size (needs at least 100–200 rows).")
    print("- Inspect df_sem.describe() for zero-variance columns.")
    print("- Drop highly correlated variables (see correlation matrix).")
    print("- Try cov_smoothing=1e-3 or higher.")


# In[ ]:


from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
import numpy as np
col_names = df_sem.columns.tolist()
cg = pc(df_sem.values, alpha=0.05, indep_test="fisherz",node_names=col_names)


# In[ ]:


print(cg.G)  # Outputs the causal graph


# In[ ]:


import pandas as pd
from dowhy import CausalModel

graph_str = """
digraph {
    Age; Academic_Pressure; Work_Pressure; CGPA; Study_Satisfaction; Job_Satisfaction;
    Sleep_Duration; Dietary_Habits; Have_you_ever_had_suicidal_thoughts_;
    Work_Study_Hours; Financial_Stress; Family_History_of_Mental_Illness;
    Gender_Male; Depression;
    Age -> Financial_Stress; Age -> Academic_Pressure; Age -> Depression; Age -> Sleep_Duration;
    Gender_Male -> Financial_Stress; Gender_Male -> Academic_Pressure; Gender_Male -> Depression;
    Gender_Male -> Sleep_Duration;
    Financial_Stress -> Academic_Pressure; Financial_Stress -> Sleep_Duration; Financial_Stress -> Depression;
    Academic_Pressure -> Sleep_Duration; Academic_Pressure -> Depression;
    Study_Satisfaction -> Academic_Pressure; Study_Satisfaction -> Depression;
    Work_Study_Hours -> Academic_Pressure; Work_Study_Hours -> Sleep_Duration; Work_Study_Hours -> Depression;
    CGPA -> Academic_Pressure; CGPA -> Study_Satisfaction;
    Work_Pressure -> Academic_Pressure;
    Sleep_Duration -> Depression;
    Dietary_Habits -> Sleep_Duration;
    Have_you_ever_had_suicidal_thoughts_ -> Depression;
    Family_History_of_Mental_Illness -> Depression; Family_History_of_Mental_Illness -> Have_you_ever_had_suicidal_thoughts_;
}
"""

# Create causal model
model = CausalModel(
    data=df_sem,
    treatment=['Work_Study_Hours'],
    outcome='Depression',
    graph=graph_str
)
model.view_model()
# Identify effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    target_units="ate"
)
print(estimate)

# # Refute estimate
# refutation = model.refute_estimate(
#     identified_estimand,
#     estimate,
#     method_name="random_common_cause"
# )
refutations = [
    model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause"),
    model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter"),
]
for ref in refutations:
    print(ref)
# print(refutation)


# In[ ]:


# Beware before running: cell output takes 90 mins

results = {}
treatments = [
    'Academic_Pressure',
    'CGPA',
    'Sleep_Duration',
    'Have_you_ever_had_suicidal_thoughts_',
    'Age',
    'Family_History_of_Mental_Illness'
]
# Iterate through each treatment
for treatment in treatments:
    print(f"\n=== Analyzing Treatment: {treatment} ===")
    
    try:
        # Initialize the causal model
        model = CausalModel(
            data=df_sem,
            treatment=[treatment],
            outcome='Depression',
            graph=graph_str
        )

        # Identify the causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate the effect
        # Use propensity score matching for binary Depression, linear regression otherwise
        estimation_method = "backdoor.linear_regression"
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=estimation_method,
            target_units="ate"
        )

        # Store the main estimate
        results[treatment] = {
            'estimated_effect': estimate.value,
            'estimand_type': identified_estimand.estimand_type,
            'method': estimation_method
        }

        # Print the estimate
        print(f"Estimand Type: {identified_estimand.estimand_type}")
        print(f"Estimation Method: {estimation_method}")
        print(f"Estimated Effect: {estimate.value}")
        print(f"Conditional Estimates:\n{estimate.conditional_estimates}")

        # Run refutation tests
        refutations = [
            model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause"),
            model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter"),
        ]

        # Store and print refutation results
        results[treatment]['refutations'] = []
        for i, refutation in enumerate(refutations, 1):
            ref_result = {
                'method': refutation.refutation_type,
                'new_effect': refutation.new_effect,
                'p_value': refutation.refutation_result['p_value'] 
            }
            results[treatment]['refutations'].append(ref_result)
            print(f"Refutation {i}: {ref_result['method']}")
            print(f"  New Effect: {ref_result['new_effect']}")
            print(f"  p-value: {ref_result['p_value']}")

    except Exception as e:
        print(f"Error analyzing {treatment}: {str(e)}")
        results[treatment] = {'error': str(e)}

# Summarize results
print("\n=== Summary of Causal Effects ===")
for treatment, result in results.items():
    if 'error' in result:
        print(f"{treatment}: Failed ({result['error']})")
    else:
        print(f"{treatment}:")
        print(f"  Estimated Effect: {result['estimated_effect']}")
        print(f"  Method: {result['method']}")
        print(f"  Refutations:")
        for ref in result['refutations']:
            print(f"    {ref['method']}: New Effect = {ref['new_effect']}, p-value = {ref['p_value']}")


# In[ ]:


# === Analyzing Treatment: Academic_Pressure === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: 0.19348120300595062 Conditional Estimates: __categorical__Dietary_Habits _categorical__Have_you_ever_had_suicidal_thoughts __categorical__Family_History_of_Mental_Illness (-1.376, -0.121] (-1.313, 0.762] (-0.969, 1.033] 0.202811 (-0.121, 1.132] (-1.313, 0.762] (-0.969, 1.033] 0.177640 (1.132, 2.386] (-1.313, 0.762] (-0.969, 1.033] 0.157418 dtype: float64 Refutation 1: Refute: Add a random common cause New Effect: 0.19348111922927125 p-value: 0.98 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Analyzing Treatment: CGPA === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: 0.010505689093822435 Conditional Estimates: __categorical__Work_Study_Hours __categorical__Dietary_Habits _categorical__Have_you_ever_had_suicidal_thoughts __categorical__Work_Pressure __categorical__Financial_Stress __categorical__Family_History_of_Mental_Illness __categorical__Gender_Male __categorical__Age (-1.932, -1.122] (-1.376, -0.121] (-1.313, 0.762] (-0.010780000000000001, 113.586] (-1.49, -0.793] (-0.969, 1.033] (-1.123, 0.891] (-1.595, -0.983] 0.012190 (-0.983, -0.371] 0.015680 (-0.371, 0.444] 0.019229 (0.444, 1.056] 0.021210 (1.056, 6.762] 0.023701 ...
# (1.036, 1.306] (-0.121, 1.132] (-1.313, 0.762] (-0.010780000000000001, 113.586] (0.599, 1.295] (-0.969, 1.033] (-1.123, 0.891] (-1.595, -0.983] -0.001061 (-0.983, -0.371] 0.003172 (-0.371, 0.444] 0.006210 (0.444, 1.056] 0.009045 (1.056, 6.762] 0.009918 Length: 212, dtype: float64 Refutation 1: Refute: Add a random common cause New Effect: 0.010505751519775229 p-value: 0.9199999999999999 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Analyzing Treatment: Sleep_Duration === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: 0.0 Conditional Estimates: _categorical__Have_you_ever_had_suicidal_thoughts __categorical__Family_History_of_Mental_Illness (-1.313, 0.762] (-0.969, 1.033] 0.0 dtype: float64 Refutation 1: Refute: Add a random common cause New Effect: -1.1102230246251566e-18 p-value: 0.98 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Analyzing Treatment: Have_you_ever_had_suicidal_thoughts_ === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: 0.26657886228551675 Conditional Estimates: Empty DataFrame Columns: [Age, Academic_Pressure, Work_Pressure, CGPA, Study_Satisfaction, Job_Satisfaction, Sleep_Duration, Dietary_Habits, Have_you_ever_had_suicidal_thoughts_, Work_Study_Hours, Financial_Stress, Family_History_of_Mental_Illness, Gender_Male, Depression, __categorical__Work_Study_Hours, __categorical__Dietary_Habits, __categorical__Work_Pressure, __categorical__Financial_Stress, __categorical__Study_Satisfaction, __categorical__Gender_Male, __categorical__Age, __categorical__CGPA, __categorical__Academic_Pressure, __categorical__Sleep_Duration] Index: []

# [0 rows x 24 columns] Refutation 1: Refute: Add a random common cause New Effect: 0.2665814067556427 p-value: 1.0 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Analyzing Treatment: Dietary_Habits === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: 0.10168217550656877 Conditional Estimates: __categorical__Work_Study_Hours _categorical__Have_you_ever_had_suicidal_thoughts __categorical__Work_Pressure __categorical__Financial_Stress __categorical__Study_Satisfaction __categorical__Family_History_of_Mental_Illness __categorical__Gender_Male __categorical__Age __categorical__CGPA __categorical__Academic_Pressure (-1.932, -1.122] (-1.313, 0.762] (-0.010780000000000001, 113.586] (-1.49, -0.793] (-2.163, -0.693] (-0.969, 1.033] (-1.123, 0.891] (-1.595, -0.983] (-5.206, -1.112] (-2.274, -0.826] 0.115965 (-0.826, -0.102] 0.096259 (-0.102, 0.622] 0.067325 (0.622, 1.346] 0.068600 (-1.112, -0.303] (-2.274, -0.826] 0.109886 ...
# (1.036, 1.306] (-1.313, 0.762] (-0.010780000000000001, 113.586] (0.599, 1.295] (0.776, 1.511] (-0.969, 1.033] (-1.123, 0.891] (1.056, 6.762] (0.345, 1.043] (-0.826, -0.102] 0.110912 (-0.102, 0.622] 0.101746 (0.622, 1.346] 0.093160 (1.043, 1.594] (-2.274, -0.826] 0.123253 (0.622, 1.346] 0.091879 Length: 7026, dtype: float64 Refutation 1: Refute: Add a random common cause New Effect: 0.10168103056807237 p-value: 0.96 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Analyzing Treatment: Age === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: -0.1102079750722329 Conditional Estimates: __categorical__Work_Study_Hours __categorical__Dietary_Habits _categorical__Have_you_ever_had_suicidal_thoughts __categorical__Work_Pressure __categorical__Study_Satisfaction __categorical__Family_History_of_Mental_Illness __categorical__Gender_Male __categorical__CGPA (-1.932, -1.122] (-1.376, -0.121] (-1.313, 0.762] (-0.010780000000000001, 113.586] (-2.163, -0.693] (-0.969, 1.033] (-1.123, 0.891] (-5.206, -1.112] -0.109140 (-1.112, -0.303] -0.108172 (-0.303, 0.345] -0.106844 (0.345, 1.043] -0.104808 (1.043, 1.594] -0.105363 ...
# (1.036, 1.306] (-0.121, 1.132] (-1.313, 0.762] (-0.010780000000000001, 113.586] (0.776, 1.511] (-0.969, 1.033] (-1.123, 0.891] (-5.206, -1.112] -0.120635 (-1.112, -0.303] -0.117191 (-0.303, 0.345] -0.115646 (0.345, 1.043] -0.112190 (1.043, 1.594] -0.113926 Length: 211, dtype: float64 Refutation 1: Refute: Add a random common cause New Effect: -0.11020922374692528 p-value: 0.96 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Analyzing Treatment: Family_History_of_Mental_Illness === Estimand Type: EstimandType.NONPARAMETRIC_ATE Estimation Method: backdoor.linear_regression Estimated Effect: 0.02633009240270834 Conditional Estimates: Empty DataFrame Columns: [Age, Academic_Pressure, Work_Pressure, CGPA, Study_Satisfaction, Job_Satisfaction, Sleep_Duration, Dietary_Habits, Have_you_ever_had_suicidal_thoughts_, Work_Study_Hours, Financial_Stress, Family_History_of_Mental_Illness, Gender_Male, Depression, __categorical__Work_Study_Hours, __categorical__Dietary_Habits, __categorical__Work_Pressure, __categorical__Financial_Stress, __categorical__Study_Satisfaction, __categorical__Gender_Male, __categorical__Age, __categorical__CGPA, __categorical__Academic_Pressure, __categorical__Sleep_Duration] Index: []

# [0 rows x 24 columns] Refutation 1: Refute: Add a random common cause New Effect: 0.02632860925233163 p-value: 0.98 Refutation 2: Refute: Use a Placebo Treatment New Effect: 0.0 p-value: 1.0

# === Summary of Causal Effects === Academic_Pressure: Estimated Effect: 0.19348120300595062 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = 0.19348111922927125, p-value = 0.98 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0 CGPA: Estimated Effect: 0.010505689093822435 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = 0.010505751519775229, p-value = 0.9199999999999999 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0 Sleep_Duration: Estimated Effect: 0.0 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = -1.1102230246251566e-18, p-value = 0.98 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0 Have_you_ever_had_suicidal_thoughts_: Estimated Effect: 0.26657886228551675 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = 0.2665814067556427, p-value = 1.0 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0 Dietary_Habits: Estimated Effect: 0.10168217550656877 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = 0.10168103056807237, p-value = 0.96 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0 Age: Estimated Effect: -0.1102079750722329 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = -0.11020922374692528, p-value = 0.96 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0 Family_History_of_Mental_Illness: Estimated Effect: 0.02633009240270834 Method: backdoor.linear_regression Refutations: Refute: Add a random common cause: New Effect = 0.02632860925233163, p-value = 0.98 Refute: Use a Placebo Treatment: New Effect = 0.0, p-value = 1.0

