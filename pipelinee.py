import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Android Games Hit Predictor",
    layout="wide"
)

st.title("🎮 Android Games Hit Prediction Dashboard")
st.write(
    "Predict whether an Android game will become a **Hit Game** "
    "using Machine Learning models."
)


# ============================================================
# LOAD + PREPROCESS DATA
# EXACTLY MATCHES THE NOTEBOOK
# ============================================================

@st.cache_data
def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv("android_games_eda_ready.csv")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Columns removed in the notebook
    cols_to_drop = [
        'game_id',
        'game_name',
        'package_name',
        'developer_name',
        'release_date',
        'soft_launch_date',
        'last_update_date',
        'featured_start_date',
        'featured_end_date',
        'row_checksum_id'
    ]

    df_prep = df.drop(
        columns=[c for c in cols_to_drop if c in df.columns]
    )

    # Separate features and target
    x = df_prep.drop(columns=['is_hit_game'])
    y = df_prep['is_hit_game']

    # Keep original feature data for the prediction interface
    original_x = x.copy()

    # ========================================================
    # EXACT NOTEBOOK STEP:
    # pd.get_dummies(x, drop_first=True)
    # ========================================================

    x_encoded = pd.get_dummies(
        x,
        drop_first=True
    )

    # Make sure boolean columns are numeric
    x_encoded = x_encoded.astype(float)

    # ========================================================
    # EXACT NOTEBOOK TRAIN/TEST SPLIT
    # ========================================================

    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # EXACT NOTEBOOK SCALING
    # ========================================================

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # ========================================================
    # EXACT NOTEBOOK PCA
    # ========================================================

    pca = PCA(
        n_components=0.95,
        random_state=42
    )

    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    return (
        df_prep,
        original_x,
        x_encoded,
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_scaled,
        x_test_scaled,
        x_train_pca,
        x_test_pca,
        scaler,
        pca
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    (
        df_prep,
        original_x,
        x_encoded,
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_scaled,
        x_test_scaled,
        x_train_pca,
        x_test_pca,
        scaler,
        pca
    ) = load_and_preprocess_data()

except Exception as e:

    st.error(f"Error loading dataset: {e}")
    st.stop()


# ============================================================
# TRAIN ALL MODELS
# EXACTLY MATCHES THE NOTEBOOK
# ============================================================

@st.cache_resource
def train_models(
    x_train_scaled,
    x_train_pca,
    y_train
):

    # ========================================================
    # EXACT NOTEBOOK MODEL DEFINITIONS
    # ========================================================

    models = {

        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        ),

        'XGBoost': XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            max_leaves=5,
            random_state=42,
            scale_pos_weight=7000 / 200
        ),

        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,
            class_weight='balanced',
            random_state=42
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),

        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42
        ),

        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),

        'Naive Bayes': GaussianNB()
    }

    trained_models = {}

    # ========================================================
    # Train models
    # ========================================================

    for name, model in models.items():

        if name == 'Naive Bayes':

            # Notebook uses PCA for Naive Bayes
            model.fit(
                x_train_pca,
                y_train
            )

        else:

            # All other models use scaled data
            model.fit(
                x_train_scaled,
                y_train
            )

        trained_models[name] = model

    return trained_models


# ============================================================
# TRAINING
# ============================================================

with st.spinner(
    "Training all model pipelines... Please wait."
):

    trained_models = train_models(
        x_train_scaled,
        x_train_pca,
        y_train
    )


# ============================================================
# CREATE EVALUATION RESULTS
# EXACTLY MATCHES NOTEBOOK METRICS
# ============================================================

def calculate_results():

    results = []

    for name, model in trained_models.items():

        # Naive Bayes uses PCA
        if name == 'Naive Bayes':

            predictions = model.predict(
                x_test_pca
            )

        else:

            predictions = model.predict(
                x_test_scaled
            )

        results.append({

            "Model": name,

            "F1-Score": round(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                4
            ),

            "Accuracy": round(
                accuracy_score(
                    y_test,
                    predictions
                ),
                4
            ),

            "Precision": round(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                4
            ),

            "Recall": round(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                4
            )
        })

    results_df = pd.DataFrame(results)

    # Same ordering as notebook comparison:
    # highest F1 first for easier presentation
    results_df = results_df.sort_values(
        by="F1-Score",
        ascending=False
    ).reset_index(drop=True)

    return results_df


results_df = calculate_results()


# ============================================================
# NAVIGATION TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "📊 Model Evaluation",
        "🔮 Make a Prediction"
    ]
)


# ============================================================
# TAB 1 — MODEL EVALUATION
# ============================================================

with tab1:

    st.header("Model Performance Metrics")

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Size",
            f"{len(df_prep):,}"
        )

    with col2:
        st.metric(
            "Training Samples",
            f"{len(x_train):,}"
        )

    with col3:
        st.metric(
            "Testing Samples",
            f"{len(x_test):,}"
        )

    with col4:
        st.metric(
            "Encoded Features",
            f"{x_encoded.shape[1]}"
        )

    st.divider()

    # --------------------------------------------------------
    # Metrics table
    # --------------------------------------------------------

    st.subheader("Model Performance")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    st.success(
        f"🏆 **Best Model: {best_model['Model']}** "
        f"with an F1-Score of **{best_model['F1-Score']:.4f}**"
    )

    # --------------------------------------------------------
    # F1 SCORE CHART
    # --------------------------------------------------------

    st.subheader("F1-Score Comparison")

    chart_df = results_df.set_index("Model")[
        ["F1-Score"]
    ]

    st.bar_chart(chart_df)

    # --------------------------------------------------------
    # PCA INFORMATION
    # --------------------------------------------------------

    st.subheader("PCA Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Original Features",
            x_train_scaled.shape[1]
        )

    with col2:

        st.metric(
            "PCA Components",
            x_train_pca.shape[1]
        )

    st.caption(
        "PCA retains 95% of the variance and is used only for "
        "the Naive Bayes model, matching the notebook."
    )


# ============================================================
# TAB 2 — CUSTOM PREDICTION
# ============================================================

with tab2:

    st.header("Predict Hit Status for a New Game")

    st.write(
        "Enter the characteristics of a new Android game "
        "and select a trained model."
    )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    selected_model_name = st.selectbox(
        "Select Model for Prediction",
        list(trained_models.keys())
    )

    selected_model = trained_models[
        selected_model_name
    ]

    st.divider()

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = {}

    # Get numerical and categorical columns
    num_cols = original_x.select_dtypes(
        include=np.number
    ).columns.tolist()

    cat_cols = original_x.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()

    col_left, col_right = st.columns(2)

    # ========================================================
    # NUMERICAL INPUTS
    # ========================================================

    for idx, col in enumerate(num_cols):

        target_col = (
            col_left
            if idx % 2 == 0
            else col_right
        )

        min_val = float(
            original_x[col].min()
        )

        max_val = float(
            original_x[col].max()
        )

        default_val = float(
            original_x[col].median()
        )

        # Prevent min/max from being identical
        if min_val == max_val:
            max_val = min_val + 1.0

        input_data[col] = target_col.number_input(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )

    # ========================================================
    # CATEGORICAL INPUTS
    # ========================================================

    for idx, col in enumerate(cat_cols):

        target_col = (
            col_left
            if idx % 2 == 0
            else col_right
        )

        options = (
            original_x[col]
            .dropna()
            .unique()
            .tolist()
        )

        options = sorted(
            options,
            key=lambda x: str(x)
        )

        if len(options) == 0:
            options = ['Unknown']

        input_data[col] = target_col.selectbox(
            f"{col}",
            options=options
        )

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    if st.button(
        "🎯 Predict Hit Status",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Create DataFrame from user input
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # EXACT SAME ENCODING AS NOTEBOOK
        # ----------------------------------------------------

        input_encoded = pd.get_dummies(
            input_df,
            drop_first=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Make input have EXACTLY the same columns as
        # x_encoded used during training.
        # ----------------------------------------------------

        input_encoded = input_encoded.reindex(
            columns=x_encoded.columns,
            fill_value=0
        )

        # Ensure same order and numeric type
        input_encoded = input_encoded.astype(float)

        # ----------------------------------------------------
        # Apply SAME scaler used for notebook-style training
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_encoded
        )

        # ----------------------------------------------------
        # Naive Bayes uses PCA
        # ----------------------------------------------------

        if selected_model_name == 'Naive Bayes':

            input_pca = pca.transform(
                input_scaled
            )

            prediction = selected_model.predict(
                input_pca
            )[0]

        else:

            prediction = selected_model.predict(
                input_scaled
            )[0]

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.write("---")

        if prediction == 1:

            st.success(
                "🎉 Prediction: **HIT GAME**"
            )

        else:

            st.error(
                "📉 Prediction: **NON-HIT GAME**"
            )

        # ----------------------------------------------------
        # Confidence / probability
        #
        # SVM in the notebook does NOT use probability=True,
        # so we do not add probability estimates here.
        # ----------------------------------------------------

        if hasattr(
            selected_model,
            "predict_proba"
        ):

            if selected_model_name == 'Naive Bayes':

                probabilities = (
                    selected_model.predict_proba(
                        input_pca
                    )[0]
                )

            else:

                probabilities = (
                    selected_model.predict_proba(
                        input_scaled
                    )[0]
                )

            hit_probability = probabilities[1]

            st.info(
                f"Model Confidence: "
                f"**{hit_probability:.2%}** probability "
                f"of being a Hit."
            )

        else:

            st.info(
                "This model does not provide probability "
                "estimates in the current configuration."
            )
