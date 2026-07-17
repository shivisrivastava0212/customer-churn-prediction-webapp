import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_pipeline():
    print("🚀 Starting model training pipeline...")
    
    # 1. Load the data statically (Updated to look for .csv)
    data_path = os.path.join('data', 'Telco_customer_churn.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing source file: Place your dataset at '{data_path}' before running.")
        
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded: {data_path} ({df.shape[0]} rows)")

    # 2. Data Cleaning: Fix 'Total Charges' (converts spaces to 0)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

    # 3. Feature Selection
    features = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents',
        'Tenure Months', 'Phone Service', 'Multiple Lines',
        'Internet Service', 'Contract', 'Paperless Billing',
        'Monthly Charges', 'Total Charges'
    ]

    X = df[features].copy()
    y = df['Churn Value']

    # 4. Fit Label Encoders dynamically for text columns
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    print("✅ Categorical variables encoded.")

    # 5. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("✅ Numerical data scaled.")

    # 6. Train Balanced Logistic Regression Model
    print("🏋️ Training balanced logistic regression model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 7. Make sure the models/ folder exists
    os.makedirs('models', exist_ok=True)

    # 8. Save artifacts (Model, Scaler, and the mapping Encoders)
    with open(os.path.join('models', 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join('models', 'encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
        
    print("🎉 Success! Artifacts saved safely inside the 'models/' directory.")

if __name__ == '__main__':
    train_pipeline()