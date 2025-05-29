# model_trainer.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

models = {}
scaler = None
label_encoder = None
df = pd.DataFrame()

def load_and_preprocess_data():
    global df
    df = pd.read_csv("Life Expectancy Data.csv")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(df[col].median())
    df.dropna(subset=['Life expectancy'], inplace=True)
    return df

def prepare_features_and_target(df):
    feature_columns = ['Year', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
        'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure',
        'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years',
        'thinness 5-9 years', 'Income composition of resources', 'Schooling']
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df['Life expectancy']
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    return X[mask], y[mask], available_features

def create_classification_target(y):
    bins = [0, 50, 65, 75, 100]
    labels = ['Very Low', 'Low', 'High', 'Very High']
    return pd.cut(y, bins=bins, labels=labels, include_lowest=True)

def display_model_performance(model_scores):
    """Display model performance metrics in a formatted way"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Separate regression and classification models
    regression_models = {k: v for k, v in model_scores.items() if v['Type'] == 'Regression'}
    classification_models = {k: v for k, v in model_scores.items() if v['Type'] == 'Classification'}
    
    # Display Regression Models
    if regression_models:
        print("\nREGRESSION MODELS:")
        print("-" * 50)
        print(f"{'Model':<18} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
        print("-" * 50)
        for model_name, scores in regression_models.items():
            print(f"{model_name:<18} {scores['RMSE']:<8.3f} {scores['MAE']:<8.3f} {scores['R2']:<8.3f}")
    
    # Display Classification Models
    if classification_models:
        print("\nCLASSIFICATION MODELS:")
        print("-" * 30)
        print(f"{'Model':<18} {'Accuracy':<10}")
        print("-" * 30)
        for model_name, scores in classification_models.items():
            print(f"{model_name:<18} {scores['Accuracy']:<10.3f}")
    
    # Find best performing models
    print("\nBEST PERFORMING MODELS:")
    print("-" * 40)
    
    if regression_models:
        best_r2_model = max(regression_models.items(), key=lambda x: x[1]['R2'])
        best_rmse_model = min(regression_models.items(), key=lambda x: x[1]['RMSE'])
        print(f"Best R² Score: {best_r2_model[0]} (R² = {best_r2_model[1]['R2']:.3f})")
        print(f"Lowest RMSE: {best_rmse_model[0]} (RMSE = {best_rmse_model[1]['RMSE']:.3f})")
    
    if classification_models:
        best_acc_model = max(classification_models.items(), key=lambda x: x[1]['Accuracy'])
        print(f"Best Accuracy: {best_acc_model[0]} (Accuracy = {best_acc_model[1]['Accuracy']:.3f})")
    
    print("="*80)

def train_all_models():
    global models, scaler, label_encoder
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    
    X, y, feature_names = prepare_features_and_target(df)
    print(f"Features used: {len(feature_names)}")
    print(f"Training samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_class = create_classification_target(y_train)
    y_test_class = create_classification_target(y_test)

    label_encoder = LabelEncoder()
    y_train_class_enc = label_encoder.fit_transform(y_train_class)
    y_test_class_enc = label_encoder.transform(y_test_class)

    model_scores = {}

    print("\nTraining models...")
    
    # Linear Regression
    print("- Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lr
    lr_pred = lr.predict(X_test_scaled)
    model_scores['Linear Regression'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'MAE': mean_absolute_error(y_test, lr_pred),
        'R2': r2_score(y_test, lr_pred),
        'Type': 'Regression'
    }

    # Random Forest
    print("- Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    rf_pred = rf.predict(X_test)
    model_scores['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'R2': r2_score(y_test, rf_pred),
        'Type': 'Regression'
    }

    # ID3 (Decision Tree Classifier)
    print("- Training ID3 (Decision Tree Classifier)...")
    id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    id3.fit(X_train, y_train_class_enc)
    models['ID3'] = id3
    id3_pred = id3.predict(X_test)
    model_scores['ID3'] = {
        'Accuracy': accuracy_score(y_test_class_enc, id3_pred),
        'Type': 'Classification'
    }

    # CART (Decision Tree Regressor)
    print("- Training CART (Decision Tree Regressor)...")
    cart = DecisionTreeRegressor(random_state=42)
    cart.fit(X_train, y_train)
    models['CART'] = cart
    cart_pred = cart.predict(X_test)
    model_scores['CART'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, cart_pred)),
        'MAE': mean_absolute_error(y_test, cart_pred),
        'R2': r2_score(y_test, cart_pred),
        'Type': 'Regression'
    }

    # Naive Bayes
    print("- Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train_class_enc)
    models['Naive Bayes'] = nb
    nb_pred = nb.predict(X_test_scaled)
    model_scores['Naive Bayes'] = {
        'Accuracy': accuracy_score(y_test_class_enc, nb_pred),
        'Type': 'Classification'
    }

    # Display performance metrics
    display_model_performance(model_scores)

    # Save everything
    print("\nSaving models...")
    os.makedirs("saved_models", exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"saved_models/{name.replace(' ', '_')}.joblib")
    joblib.dump(scaler, "saved_models/scaler.joblib")
    joblib.dump(label_encoder, "saved_models/label_encoder.joblib")
    joblib.dump(model_scores, "saved_models/model_scores.joblib")
    joblib.dump(feature_names, "saved_models/feature_names.joblib")

    print("Models trained and saved successfully!")

if __name__ == "__main__":
    train_all_models()
