import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Android Games Hit Predictor", layout="wide")

st.title("🎮 Android Games Hit Prediction Dashboard")
st.write("Predict whether an Android game will become a **Hit Game** using Machine Learning pipelines.")

# Load raw dataset strictly for metadata and range values (UI inputs)
@st.cache_data
def load_metadata():
    df = pd.read_csv("android_games_eda_ready.csv")
    df.drop_duplicates(inplace=True)
    
    cols_to_drop = [
        'game_id', 'game_name', 'package_name', 'developer_name',
        'release_date', 'soft_launch_date', 'last_update_date',
        'featured_start_date', 'featured_end_date', 'row_checksum_id'
    ]
    df_prep = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    X = df_prep.drop(columns=['is_hit_game'])
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return X, num_cols, cat_cols

try:
    X, num_cols, cat_cols = load_metadata()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Load pre-trained models exported from the notebook
@st.cache_resource
def load_trained_pipelines():
    return {
        'Decision Tree': joblib.load('models/decision_tree.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting.pkl'),
        'XGBoost': joblib.load('models/xgboost.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'SVM': joblib.load('models/svm.pkl'),
        'KNN': joblib.load('models/knn.pkl'),
        'Naive Bayes (PCA)': joblib.load('models/naive_bayes_pca.pkl')
    }

trained_models = load_trained_pipelines()

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Model Evaluation", "🔮 Make a Prediction"])

# TAB 1: Notebook Metrics (Exact match)
with tab1:
    st.header("Model Performance Metrics")
    
    # Exact results calculated directly from your test evaluation set
    metrics_data = [
        {"Model": "Decision Tree", "F1-Score": 0.9620, "Accuracy": 0.9985, "Precision": 0.9744, "Recall": 0.9500},
        {"Model": "Gradient Boosting", "F1-Score": 0.9041, "Accuracy": 0.9965, "Precision": 1.0000, "Recall": 0.8250},
        {"Model": "XGBoost", "F1-Score": 0.8791, "Accuracy": 0.9945, "Precision": 0.7843, "Recall": 1.0000},
        {"Model": "Random Forest", "F1-Score": 0.8000, "Accuracy": 0.9920, "Precision": 0.8000, "Recall": 0.8000},
        {"Model": "SVM", "F1-Score": 0.6506, "Accuracy": 0.9855, "Precision": 0.6279, "Recall": 0.6750},
        {"Model": "KNN", "F1-Score": 0.5263, "Accuracy": 0.9865, "Precision": 0.8824, "Recall": 0.3750},
        {"Model": "Naive Bayes (PCA)", "F1-Score": 0.3600, "Accuracy": 0.9680, "Precision": 0.3000, "Recall": 0.4500}
    ]
    
    results_df = pd.DataFrame(metrics_data).sort_values(by="F1-Score", ascending=False)
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
    
    # Render numerical inputs
    for idx, col in enumerate(num_cols):
        target_col = col_left if idx % 2 == 0 else col_right
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        default_val = float(X[col].median())
        
        if min_val == max_val:
            max_val = min_val + 1.0
            
        input_data[col] = target_col.number_input(f"{col}", value=default_val, min_value=min_val, max_value=max_val)
        
    # Render categorical inputs
    for idx, col in enumerate(cat_cols):
        target_col = col_left if idx % 2 == 0 else col_right
        options = X[col].dropna().unique().tolist()
        if not options:
            options = ['Unknown']
        input_data[col] = target_col.selectbox(f"{col}", options=options)
        
    if st.button("Predict Hit Status", type="primary"):
        input_df = pd.DataFrame([input_data])
        
        # Predict using loaded notebook pipeline directly
        prediction = selected_model.predict(input_df)[0]
        
        st.write("---")
        if prediction == 1:
            st.success("🎉 Prediction: **HIT GAME**")
        else:
            st.error("📉 Prediction: **NON-HIT GAME**")
        
        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(input_df)[0]
            st.info(f"Model Confidence: **{probs[1]:.2%}** probability of being a Hit.")
