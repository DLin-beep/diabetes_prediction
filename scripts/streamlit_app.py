import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import os

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes.csv')
    df = pd.read_csv(data_path)
    return df

df = load_data()

st.title('Diabetes Classification with Logistic Regression')
st.write('Predict diabetes outcome using Logistic Regression on the diabetes dataset.')

st.subheader('Data Preview')
st.dataframe(df.head())

columns = df.columns.tolist()
target_col = st.selectbox('Select target column (should be binary)', columns, index=len(columns)-1)
feature_cols = st.multiselect('Select feature columns', [col for col in columns if col != target_col], default=[col for col in columns if col != target_col])

if len(feature_cols) < 2:
    st.warning('Please select at least two feature columns for 3D plot.')
    st.stop()
x_axis_feature = st.selectbox('Select X-axis feature for 3D plot', feature_cols, index=0)
y_axis_feature = st.selectbox('Select Y-axis feature for 3D plot', [col for col in feature_cols if col != x_axis_feature], index=0)

st.subheader('Model Hyperparameters')
C = st.slider('Inverse Regularization Strength (C)', 0.01, 10.0, 1.0, 0.01)
penalty = st.selectbox('Penalty', ['l2', 'l1', 'elasticnet', 'none'])
solver = st.selectbox('Solver', ['lbfgs', 'liblinear', 'saga'])
l1_ratio = None
if penalty == 'elasticnet':
    l1_ratio = st.slider('L1 Ratio (ElasticNet only)', 0.0, 1.0, 0.5, 0.01)

st.subheader('Train/Test Split')
test_size = st.slider('Test set size (%)', 10, 50, 20, 5) / 100

if st.button('Run Classification'):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio if penalty == 'elasticnet' else None, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader('Classification Metrics')
    st.write(f'Accuracy: {acc:.4f}')
    st.subheader('Confusion Matrix')
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)
    st.subheader(f'3D Plot: {x_axis_feature} vs {y_axis_feature} vs Predicted Probability')
    x_vals = X_test[x_axis_feature].values
    y_vals = X_test[y_axis_feature].values
    z_vals = y_proba
    actual_class = y_test.values
    x_range = np.linspace(X_test[x_axis_feature].min(), X_test[x_axis_feature].max(), 30)
    y_range = np.linspace(X_test[y_axis_feature].min(), X_test[y_axis_feature].max(), 30)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = pd.DataFrame({x_axis_feature: xx.ravel(), y_axis_feature: yy.ravel()})
    for col in feature_cols:
        if col not in [x_axis_feature, y_axis_feature]:
            grid[col] = X_test[col].mean()
    zz = model.predict_proba(grid[feature_cols])[:, 1].reshape(xx.shape)
    trace0 = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=6,
            color=actual_class,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Actual Class')
        ),
        text=[f'Actual: {a}' for a in actual_class],
        name='Test Points'
    )
    trace1 = go.Surface(
        x=x_range,
        y=y_range,
        z=zz,
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name='Prediction Surface'
    )
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=x_axis_feature),
            yaxis=dict(title=y_axis_feature),
            zaxis=dict(title='Predicted Probability')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700,
        title=f'3D Scatter: {x_axis_feature} vs {y_axis_feature} vs Predicted Probability'
    )
    fig3d = go.Figure(data=[trace1, trace0], layout=layout)
    st.plotly_chart(fig3d, use_container_width=True)
    st.subheader('ROC Curve')
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc) 