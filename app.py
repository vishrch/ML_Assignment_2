import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Application configuration
st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write(
    "Compare five machine learning classifiers to predict "
    "whether a customer subscribes to a term deposit."
)

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Dataset upload
st.sidebar.header("Evaluation Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file, sep=None, engine="python")

    if "y" not in test_data.columns:
        st.error("The uploaded CSV must contain the target column 'y'.")
        st.stop()

    X_test = test_data.drop(columns=["y"])
    y_test = test_data["y"].map({"no": 0, "yes": 1})

    if y_test.isnull().any():
        st.error("The target column must contain only 'yes' or 'no'.")
        st.stop()

    model_path = MODEL_DIR / MODEL_FILES[selected_model]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader(f"Results: {selected_model}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(cm_df)

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Subscription", "Subscription"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    # Preview uploaded dataset
    with st.expander("View Uploaded Dataset"):
        st.dataframe(test_data.head(20))

else:

    st.info(
        "Upload the test_data.csv file to evaluate the selected model."
    )

    st.subheader("Available Models")

    for model_name in MODEL_FILES:
        st.write(f"• {model_name}")