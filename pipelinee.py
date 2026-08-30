import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Android Games Hit Predictor", layout="wide")

st.title("🎮 Android Games Hit Predictor")
st.write("Predict whether an Android game will be a hit using machine learning.")

@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("android_games_eda_ready.csv")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values (Categorical -> 'Unknown', Numerical -> Median)
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=np.number).columns
    
    df[cat_cols] = df[cat_cols].fillna('Unknown').astype(str)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    return df

# Load data
df = load_and_preprocess_data()

# Define features and target
target_col = 'is_hit_game'

# Select core numerical features for modeling & user input
feature_cols = [
    'price_usd', 'downloads', 'retention_day1_pct', 'retention_day7_pct',
    'retention_day30_pct', 'active_users_30d', 'avg_session_minutes',
    'avg_daily_sessions', 'crash_rate_pct', 'apk_size_mb',
    'update_frequency_days', 'rating_avg', 'rating_count', 'review_count',
    'marketing_spend_usd', 'cpi_usd', 'conversion_to_payer_pct',
    'ad_impressions_per_user', 'arpu_usd', 'total_revenue_usd'
]

X = df[feature_cols]
y = df[target_col]

# Train-test split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train_scaled, y_train)

# User Input Interface in Streamlit
st.sidebar.header("Input Game Features")

user_inputs = {}
for col in feature_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    default_val = float(df[col].median())
    user_inputs[col] = st.sidebar.number_input(
        f"{col}", min_value=min_val, max_value=max_val, value=default_val
    )

input_df = pd.DataFrame([user_inputs])
scaled_input = scaler.transform(input_df)

# Prediction Logic
if st.button("Predict Game Status"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"🎉 **HIT GAME!** (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ **NON-HIT GAME** (Probability of being a hit: {probability:.2%})")