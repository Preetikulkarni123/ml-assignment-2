"""
Streamlit App for ML Assignment 2
Interactive web application for classification model demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
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
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤖 ML Classification Model Demo</div>', unsafe_allow_html=True)
st.markdown("### BITS Pilani - Machine Learning Assignment 2")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=200)
    st.markdown("---")
    st.markdown("## 📊 Application Guide")
    st.markdown("""
    1. **Upload Test Data** (CSV format)
    2. **Select Model** from dropdown
    3. **View Metrics** and results
    4. **Analyze** confusion matrix
    """)
    st.markdown("---")
    st.info("💡 **Note**: Upload only test data due to Streamlit free tier limits")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Upload", "🔮 Model Prediction", "📈 Results", "ℹ️ About"])

# ============================================================================
# TAB 1: DATA UPLOAD
# ============================================================================
with tab1:
    st.header("📤 Upload Test Dataset")
    st.markdown("Upload your test dataset in CSV format (without target variable)")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your test data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            test_data = pd.read_csv(uploaded_file)
            st.session_state['test_data'] = test_data
            
            st.success(f"✅ File uploaded successfully!")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", test_data.shape[0])
            with col2:
                st.metric("Total Columns", test_data.shape[1])
            with col3:
                st.metric("Missing Values", test_data.isnull().sum().sum())
            
            # Show data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(test_data.head(10), use_container_width=True)
            
            # Show statistics
            with st.expander("📊 View Statistical Summary"):
                st.dataframe(test_data.describe(), use_container_width=True)
            
            # Show data types
            with st.expander("🔍 View Data Types"):
                dtype_df = pd.DataFrame({
                    'Column': test_data.columns,
                    'Data Type': test_data.dtypes.values,
                    'Non-Null Count': test_data.count().values
                })
                st.dataframe(dtype_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin")

# ============================================================================
# TAB 2: MODEL PREDICTION
# ============================================================================
with tab2:
    st.header("🔮 Model Selection & Prediction")
    
    if 'test_data' in st.session_state:
        
        # Model selection dropdown
        st.markdown("### Select Classification Model")
        
        model_options = {
            "Logistic Regression": "model_lr.pkl",
            "Decision Tree": "model_dt.pkl",
            "K-Nearest Neighbors (kNN)": "model_knn.pkl",
            "Naive Bayes": "model_nb.pkl",
            "Random Forest": "model_rf.pkl",
            "XGBoost": "model_xgb.pkl"
        }
        
        selected_model_name = st.selectbox(
            "Choose a model:",
            list(model_options.keys()),
            help="Select the classification model you want to use"
        )
        
        st.markdown("---")
        
        # Target variable input
        col1, col2 = st.columns(2)
        with col1:
            target_column = st.text_input(
                "Target Column Name",
                placeholder="e.g., target, label, class",
                help="Enter the name of your target variable column"
            )
        
        with col2:
            st.markdown("### Model Info")
            model_descriptions = {
                "Logistic Regression": "Linear model for binary/multiclass classification",
                "Decision Tree": "Tree-based model using if-then-else rules",
                "K-Nearest Neighbors (kNN)": "Instance-based learning using proximity",
                "Naive Bayes": "Probabilistic classifier based on Bayes theorem",
                "Random Forest": "Ensemble of decision trees",
                "XGBoost": "Gradient boosting ensemble method"
            }
            st.info(model_descriptions[selected_model_name])
        
        # Prediction button
        if st.button("🚀 Run Prediction & Evaluation", type="primary"):
            if not target_column:
                st.error("❌ Please enter the target column name")
            elif target_column not in st.session_state['test_data'].columns:
                st.error(f"❌ Column '{target_column}' not found in dataset")
            else:
                try:
                    with st.spinner("🔄 Loading model and making predictions..."):
                        
                        # Load model and scaler
                        model = joblib.load(f'models/{model_options[selected_model_name]}')
                        scaler = joblib.load('models/scaler.pkl')
                        
                        test_data = st.session_state['test_data']
                        
                        # Separate features and target
                        X_test = test_data.drop(columns=[target_column])
                        y_test = test_data[target_column]
                        
                        # Handle categorical variables
                        X_test = pd.get_dummies(X_test, drop_first=True)
                        
                        # Feature scaling
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        mcc = matthews_corrcoef(y_test, y_pred)
                        
                        # Calculate AUC
                        try:
                            y_pred_proba = model.predict_proba(X_test_scaled)
                            if len(np.unique(y_test)) > 2:
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                            else:
                                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        except:
                            auc = None
                        
                        # Store results in session state
                        st.session_state['metrics'] = {
                            'Accuracy': accuracy,
                            'AUC': auc,
                            'Precision': precision,
                            'Recall': recall,
                            'F1 Score': f1,
                            'MCC': mcc
                        }
                        st.session_state['y_test'] = y_test
                        st.session_state['y_pred'] = y_pred
                        st.session_state['model_name'] = selected_model_name
                        
                        st.success("✅ Prediction completed successfully!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.info("💡 Make sure you've trained models and saved them in the 'models' folder")
    else:
        st.warning("⚠️ Please upload test data first in the 'Data Upload' tab")

# ============================================================================
# TAB 3: RESULTS
# ============================================================================
with tab3:
    st.header("📈 Model Performance Results")
    
    if 'metrics' in st.session_state:
        
        st.markdown(f"### Results for: **{st.session_state['model_name']}**")
        st.markdown("---")
        
        # Display metrics in cards
        st.markdown("### 📊 Evaluation Metrics")
        
        col1, col2, col3 = st.columns(3)
        metrics = st.session_state['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        
        with col2:
            if metrics['AUC'] is not None:
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
            else:
                st.metric("AUC Score", "N/A")
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            st.metric("MCC Score", f"{metrics['MCC']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {st.session_state["model_name"]}', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("### 📋 Classification Report")
        
        report = classification_report(
            st.session_state['y_test'], 
            st.session_state['y_pred'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Download results
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        results_df = pd.DataFrame([metrics])
        csv = results_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Metrics as CSV",
            data=csv,
            file_name=f"{st.session_state['model_name']}_metrics.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👆 Please run predictions in the 'Model Prediction' tab first")

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎓 ML Assignment 2 - BITS Pilani
    
    This interactive web application demonstrates the implementation and comparison of 
    **6 different classification algorithms** for machine learning.
    
    #### 📚 Models Implemented:
    1. **Logistic Regression** - Linear classification model
    2. **Decision Tree** - Tree-based classifier
    3. **K-Nearest Neighbors** - Instance-based learning
    4. **Naive Bayes** - Probabilistic classifier
    5. **Random Forest** - Ensemble method (Bagging)
    6. **XGBoost** - Gradient boosting ensemble
    
    #### 📊 Evaluation Metrics:
    - **Accuracy**: Overall correctness of predictions
    - **AUC**: Area Under ROC Curve
    - **Precision**: Positive predictive value
    - **Recall**: Sensitivity or True Positive Rate
    - **F1 Score**: Harmonic mean of precision and recall
    - **MCC**: Matthews Correlation Coefficient
    
    #### 🛠️ Technology Stack:
    - **Framework**: Streamlit
    - **ML Libraries**: scikit-learn, XGBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    
    #### 👨‍💻 Developer Information:
    - **Course**: M.Tech (AIML/DSE) - Machine Learning
    - **Assignment**: Assignment 2
    - **Institute**: BITS Pilani
    
    ---
    
    #### 📝 Instructions for Use:
    1. Upload your test dataset (CSV format)
    2. Select a classification model
    3. Enter the target column name
    4. Click "Run Prediction & Evaluation"
    5. View comprehensive results and metrics
    
    #### ⚠️ Important Notes:
    - Upload **only test data** due to Streamlit free tier limitations
    - Ensure your CSV file is properly formatted
    - Target column should be included in the uploaded data
    - Models should be pre-trained and saved in 'models' folder
    
    ---
    
    ### 🔗 Useful Resources:
    - [Scikit-learn Documentation](https://scikit-learn.org/)
    - [XGBoost Documentation](https://xgboost.readthedocs.io/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    
    ---
    
    **Made with ❤️ for BITS Pilani ML Assignment**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2026 BITS Pilani | M.Tech AIML/DSE | Machine Learning Assignment 2"
    "</div>",
    unsafe_allow_html=True
)
