import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Android Games Hit Predictor", layout="wide")

st.title("🎮 Android Games Hit Prediction Dashboard")
st.write("Predict whether an Android game will become a **Hit Game** using Machine Learning pipelines.")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("android_games_eda_ready.csv")
    df.drop_duplicates(inplace=True)
    
    cols_to_drop = [
        'game_id', 'game_name', 'package_name', 'developer_name',
        'release_date', 'soft_launch_date', 'last_update_date',
        'featured_start_date', 'featured_end_date', 'row_checksum_id'
    ]
    df_prep = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    X = df_prep.drop(columns=['is_hit_game'])
    y = df_prep['is_hit_game']
    
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Stratified split to preserve class imbalance ratio across sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X, y, X_train, X_test, y_train, y_test, num_cols, cat_cols

try:
    X, y, X_train, X_test, y_train, y_test, num_cols, cat_cols = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Helper constructor for preprocessor
def build_preprocessor():
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

@st.cache_resource
def train_models(X_tr, y_tr):
    negative_count = (y_tr == 0).sum()
    positive_count = (y_tr == 1).sum()
    dynamic_scale = negative_count / positive_count if positive_count > 0 else 1.0

    model_dict = {
        'XGBoost': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, max_leaves=5, random_state=42, scale_pos_weight=dynamic_scale))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42))
        ]),
        'SVM': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True))
        ]),
        # Naive Bayes pipeline with PCA to handle multi-collinearity
        'Naive Bayes (PCA)': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('classifier', GaussianNB())
        ]),
        # Unweighted default KNN in high dimensions yields lowest F1 score on imbalanced target
        'KNN': Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])
    }
    
    trained = {}
    for name, pipe in model_dict.items():
        pipe.fit(X_tr, y_tr)
        trained[name] = pipe
    return trained

with st.spinner("Training all model pipelines... Please wait."):
    trained_models = train_models(X_train, y_train)

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Model Evaluation", "🔮 Make a Prediction"])

# TAB 1: Evaluation Metrics
with tab1:
    st.header("Model Performance Metrics")
    
    results = []
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "F1-Score": round(f1_score(y_test, preds, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, zero_division=0), 4)
        })
    
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    st.dataframe(results_df, use_container_width=True)
    
    st.subheader("F1-Score Comparison")
    st.bar_chart(results_df.set_index("Model")["F1-Score"])

# TAB 2: Custom Prediction Input
with tab2:
    st.header("Predict Hit Status for New Game")
    
    selected_model_name = st.selectbox("Select Model for Prediction", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]
    
    input_data = {}
    col_left, col_right = st.columns(2)
    
    # Render numerical feature inputs
    for idx, col in enumerate(num_cols):
        target_col = col_left if idx % 2 == 0 else col_right
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        default_val = float(X[col].median())
        
        # Handle constant features safely
        if min_val == max_val:
            max_val = min_val + 1.0
            
        input_data[col] = target_col.number_input(f"{col}", value=default_val, min_value=min_val, max_value=max_val)
        
    # Render categorical feature inputs
    for idx, col in enumerate(cat_cols):
        target_col = col_left if idx % 2 == 0 else col_right
        options = X[col].dropna().unique().tolist()
        if not options:
            options = ['Unknown']
        input_data[col] = target_col.selectbox(f"{col}", options=options)
        
    if st.button("Predict Hit Status", type="primary"):
        input_df = pd.DataFrame([input_data])
        prediction = selected_model.predict(input_df)[0]
        
        st.write("---")
        if prediction == 1:
            st.success("🎉 Prediction: **HIT GAME**")
        else:
            st.error("📉 Prediction: **NON-HIT GAME**")
        
        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(input_df)[0]
            st.info(f"Model Confidence: **{probs[1]:.2%}** probability of being a Hit.")
