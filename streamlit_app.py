import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import additional models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Hair Fall Prediction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    
    .info-card h4, .info-card h5 {
        color: #1e3a8a;
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .info-card p, .info-card li {
        color: #374151;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Risk level styling */
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Recommendation box */
    .recommendation {
        background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #8b5cf6;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom selectbox */
    .stSelectbox > div > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Set custom color palettes
MAIN_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
RISK_COLORS = {'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#10b981'}
HAIR_LOSS_COLORS = {'Hair Loss': '#ef4444', 'No Hair Loss': '#10b981'}

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('data/Predict Hair Fall.csv')
    df.columns = df.columns.str.strip()
    df = df.replace('No Data', np.nan)
    return df

# Prepare data for ML
@st.cache_data
def prepare_ml_data(df):
    df_ml = df.copy().dropna()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['Medical Conditions', 'Medications & Treatments', 'Nutritional Deficiencies']
    
    for col in categorical_features:
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le
    
    # Binary encoding
    binary_cols = ['Genetics', 'Hormonal Changes', 'Poor Hair Care Habits', 
                   'Environmental Factors', 'Smoking', 'Weight Loss']
    for col in binary_cols:
        df_ml[col + '_encoded'] = df_ml[col].map({'Yes': 1, 'No': 0})
    
    # Encode stress levels
    stress_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    df_ml['Stress_encoded'] = df_ml['Stress'].map(stress_map)
    
    feature_cols = (['Age', 'Stress_encoded'] + [col + '_encoded' for col in binary_cols] + 
                   [col + '_encoded' for col in categorical_features])
    
    X = df_ml[feature_cols]
    y = df_ml['Hair Loss']
    
    return X, y, feature_cols, label_encoders, binary_cols

# Train improved models with target encoding and enhanced model selection
@st.cache_resource
def train_models(df):
    # Proper preprocessing with target encoding
    df_processed = df.copy().dropna()
    
    # Binary features
    binary_features = ['Genetics', 'Hormonal Changes', 'Poor Hair Care Habits', 
                       'Environmental Factors', 'Smoking', 'Weight Loss']
    for feature in binary_features:
        df_processed[feature] = df_processed[feature].map({'Yes': 1, 'No': 0})
    
    # Ordinal encoding for stress
    stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
    df_processed['Stress'] = df_processed['Stress'].map(stress_mapping)
    
    # Target encoding for categorical features
    categorical_features = ['Medical Conditions', 'Medications & Treatments', 'Nutritional Deficiencies']
    for feature in categorical_features:
        target_mean = df_processed.groupby(feature)['Hair Loss'].mean()
        df_processed[f'{feature}_target_encoded'] = df_processed[feature].map(target_mean)
    
    # Prepare features
    feature_columns = (['Age', 'Stress'] + binary_features + 
                      [f'{feature}_target_encoded' for feature in categorical_features])
    
    X = df_processed[feature_columns]
    y = df_processed['Hair Loss']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=271, max_depth=10, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=11, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
    
    results = {}
    for name, model in models.items():
        try:
            # Use scaled data for logistic regression, regular data for tree-based models
            if 'Logistic' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                accuracy = model.score(X_test_scaled, y_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                accuracy = model.score(X_test, y_test)
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'scaler': scaler if 'Logistic' in name else None
            }
        except Exception as e:
            st.warning(f"Could not train {name}: {str(e)}")
    
    return results, X_test, y_test, feature_columns, categorical_features, binary_features

def main():
    st.markdown('<h1 class="main-header">Hair Fall Prediction Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar navigation with enhanced styling
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Choose Analysis", [
        "Dataset Overview", 
        "Risk Assessment Tool",
        "Data Visualizations", 
        "ML Model Performance",
        "Cluster Analysis"
    ])
    
    if page == "Dataset Overview":
        show_dataset_overview(df)
    elif page == "Risk Assessment Tool":
        show_risk_assessment(df)
    elif page == "Data Visualizations":
        show_visualizations(df)
    elif page == "ML Model Performance":
        show_ml_performance(df)
    elif page == "Cluster Analysis":
        show_cluster_analysis(df)

def show_dataset_overview(df):
    st.header("Dataset Overview")
    
    # Enhanced metrics with custom styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{len(df)}</h3>
            <p style="margin:0; font-size: 0.9rem;">Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{df['Hair Loss'].mean():.1%}</h3>
            <p style="margin:0; font-size: 0.9rem;">Hair Loss Rate</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{df['Age'].mean():.1f}</h3>
            <p style="margin:0; font-size: 0.9rem;">Average Age</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{len(df.columns) - 1}</h3>
            <p style="margin:0; font-size: 0.9rem;">Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Age Distribution
    st.subheader("Age Distribution by Hair Loss Status")
    df_temp = df.copy()
    df_temp['Age_Group'] = pd.cut(df_temp['Age'], bins=[0, 25, 35, 45, 100], 
                                labels=['18-25', '26-35', '36-45', '46+'])
    df_temp['Hair_Loss_Status'] = df_temp['Hair Loss'].map({1: 'Hair Loss', 0: 'No Hair Loss'})
    
    fig = px.histogram(
        df_temp, 
        x='Age_Group', 
        color='Hair_Loss_Status',
        title='Hair Loss Distribution Across Age Groups',
        labels={'Hair_Loss_Status': 'Status', 'count': 'Number of Patients'},
        color_discrete_map=HAIR_LOSS_COLORS,
        template='plotly_white'
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        title_font=dict(size=16, color='#1e3a8a'),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Medical Conditions with enhanced styling
        st.subheader("Most Common Medical Conditions")
        medical_conditions = df['Medical Conditions'].value_counts().head(8)
        
        fig = px.bar(
            x=medical_conditions.values, 
            y=medical_conditions.index,
            orientation='h',
            title='Top Medical Conditions',
            labels={'x': 'Number of Patients', 'y': 'Medical Condition'},
            color=medical_conditions.values,
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=10),
            title_font=dict(size=14, color='#1e3a8a'),
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress Level Distribution
        st.subheader("Stress Level Distribution")
        stress_counts = df['Stress'].value_counts()
        
        fig = px.pie(
            values=stress_counts.values,
            names=stress_counts.index,
            title='Stress Levels in Population',
            color_discrete_sequence=MAIN_COLORS,
            template='plotly_white'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=14, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factors Overview
    st.subheader("Risk Factors Overview")
    risk_factors = ['Genetics', 'Poor Hair Care Habits', 'Smoking', 'Environmental Factors']
    risk_data = []
    
    for factor in risk_factors:
        yes_rate = (df[factor] == 'Yes').mean()
        risk_data.append({'Factor': factor, 'Prevalence': yes_rate})
    
    risk_df = pd.DataFrame(risk_data)
    
    fig = px.bar(
        risk_df, 
        x='Factor', 
        y='Prevalence',
        title='Prevalence of Key Risk Factors',
        labels={'Prevalence': 'Prevalence Rate', 'Factor': 'Risk Factor'},
        color='Prevalence',
        color_continuous_scale='Reds',
        template='plotly_white'
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        title_font=dict(size=16, color='#1e3a8a'),
        showlegend=False,
        coloraxis_showscale=False
    )
    fig.update_traces(text=risk_df['Prevalence'].apply(lambda x: f'{x:.1%}'), textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment(df):
    st.header("Personal Hair Loss Risk Assessment")
    
    st.markdown("""
    <div class="info-card">
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">
            Enter your personal information below to receive a comprehensive risk assessment 
            using our advanced machine learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get trained models
    results, _, _, feature_columns, categorical_features, binary_features = train_models(df)
    
    # Use the best performing model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    scaler = results[best_model_name]['scaler']
    
    st.info(f"Using **{best_model_name}** (AUC: {results[best_model_name]['auc']:.3f})")
    
    # Create input form with better styling
    with st.container():
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            age = st.slider("Age", 18, 70, 30, help="Your current age")
            stress = st.selectbox("Stress Level", ['Low', 'Moderate', 'High'], 
                                help="Your current stress level")
            
            st.markdown("**Lifestyle Factors**")
            genetics = st.selectbox("Genetic Predisposition", ['No', 'Yes'], 
                                  help="Family history of hair loss")
            smoking = st.selectbox("Smoking", ['No', 'Yes'], 
                                 help="Do you currently smoke?")
            poor_hair_care = st.selectbox("Poor Hair Care Habits", ['No', 'Yes'], 
                                        help="Excessive heat styling, harsh chemicals, etc.")
        
        with col2:
            st.markdown("**Health Factors**")
            hormonal_changes = st.selectbox("Hormonal Changes", ['No', 'Yes'], 
                                          help="Recent hormonal changes or imbalances")
            environmental_factors = st.selectbox("Environmental Factors", ['No', 'Yes'], 
                                                help="Exposure to pollution, chemicals, etc.")
            weight_loss = st.selectbox("Recent Weight Loss", ['No', 'Yes'], 
                                     help="Significant weight loss in recent months")
            
            st.markdown("**Medical Information**")
            medical_condition = st.selectbox("Medical Condition", 
                                           sorted(df['Medical Conditions'].dropna().unique()),
                                           help="Select your primary medical condition")
            medication = st.selectbox("Current Medication/Treatment", 
                                    sorted(df['Medications & Treatments'].dropna().unique()),
                                    help="Your current medication or treatment")
            nutrition_def = st.selectbox("Nutritional Deficiency", 
                                       sorted(df['Nutritional Deficiencies'].dropna().unique()),
                                       help="Known nutritional deficiencies")
    
    st.markdown("---")
    
    # Enhanced prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Assess My Risk", type="primary", use_container_width=True):
            try:
                # Prepare input data for prediction
                input_dict = {
                    'Age': age,
                    'Stress': {'Low': 1, 'Moderate': 2, 'High': 3}[stress],
                    'Genetics': 1 if genetics == 'Yes' else 0,
                    'Hormonal Changes': 1 if hormonal_changes == 'Yes' else 0,
                    'Poor Hair Care Habits': 1 if poor_hair_care == 'Yes' else 0,
                    'Environmental Factors': 1 if environmental_factors == 'Yes' else 0,
                    'Smoking': 1 if smoking == 'Yes' else 0,
                    'Weight Loss': 1 if weight_loss == 'Yes' else 0
                }
                
                # Add target encoded features
                df_for_encoding = df.copy().dropna()
                for feature in categorical_features:
                    target_mean_map = df_for_encoding.groupby(feature)['Hair Loss'].mean().to_dict()
                    if feature == 'Medical Conditions':
                        input_dict[f'{feature}_target_encoded'] = target_mean_map.get(medical_condition, 0.5)
                    elif feature == 'Medications & Treatments':
                        input_dict[f'{feature}_target_encoded'] = target_mean_map.get(medication, 0.5)
                    elif feature == 'Nutritional Deficiencies':
                        input_dict[f'{feature}_target_encoded'] = target_mean_map.get(nutrition_def, 0.5)
                
                # Create input dataframe
                input_data = pd.DataFrame([input_dict])
                input_data = input_data[feature_columns]  # Ensure correct column order
                
                # Make prediction
                if scaler:  # For logistic regression
                    input_data_scaled = scaler.transform(input_data)
                    risk_prob = best_model.predict_proba(input_data_scaled)[0][1]
                else:  # For tree-based models
                    risk_prob = best_model.predict_proba(input_data)[0][1]
                
                # Determine risk level
                if risk_prob > 0.6:
                    risk_level = "High"
                    risk_color = "risk-high"
                elif risk_prob > 0.45:
                    risk_level = "Moderate"
                    risk_color = "risk-moderate"
                else:
                    risk_level = "Low"
                    risk_color = "risk-low"
                
                # Display results with enhanced styling
                st.markdown("### Your Risk Assessment Results")
                
                # Results metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="{risk_color}">
                        <h3 style="margin:0; font-size: 2rem;">{risk_prob:.1%}</h3>
                        <p style="margin:0;">Risk Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="{risk_color}">
                        <h3 style="margin:0; font-size: 2rem;">{risk_level}</h3>
                        <p style="margin:0;">Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    confidence = "High" if abs(risk_prob - 0.5) > 0.1 else "Moderate"
                    st.markdown(f"""
                    <div class="{risk_color}">
                        <h3 style="margin:0; font-size: 2rem;">{confidence}</h3>
                        <p style="margin:0;">Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Personalized recommendations
                st.markdown("### Personalized Recommendations")
                recommendations = []
                
                if age >= 26 and age <= 35:
                    recommendations.append("**Early Screening**: You're in the highest risk age group - consider regular monitoring")
                if stress in ['Moderate', 'High']:
                    recommendations.append("**Stress Management**: Implement stress reduction techniques (meditation, exercise, therapy)")
                if poor_hair_care == 'Yes':
                    recommendations.append("**Hair Care**: Improve routine - avoid excessive heat styling and harsh chemicals")
                if nutrition_def != 'No Data' and 'Deficiency' not in nutrition_def:
                    recommendations.append(f"**Nutrition**: Address {nutrition_def.lower()} through supplements or dietary changes")
                if smoking == 'Yes':
                    recommendations.append("**Lifestyle**: Consider smoking cessation for overall hair and health benefits")
                if genetics == 'Yes':
                    recommendations.append("**Genetic Factor**: Consider early intervention and professional consultation")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation">
                            • {rec}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="recommendation">
                        <strong>Good News!</strong> Your risk factors are well-managed. Continue with:
                        <br>• Regular health check-ups
                        <br>• Balanced nutrition
                        <br>• Healthy lifestyle maintenance
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model explanation
                st.markdown("### How We Calculate Your Risk")
                st.info(f"""
                Your risk assessment is based on a **{best_model_name}** model trained on {len(df)} patient records. 
                The model achieved an AUC score of **{results[best_model_name]['auc']:.3f}**, indicating moderate predictive capability.
                
                **Key factors influencing your assessment:**
                - Age and genetic predisposition
                - Lifestyle factors (stress, smoking, hair care)
                - Medical conditions and treatments
                - Nutritional status
                
                **Important:** This is a screening tool and should not replace professional medical advice.
                """)
                
            except Exception as e:
                st.error(f"Error processing your assessment: {str(e)}")
                st.error("Please check your inputs and try again.")

def show_visualizations(df):
    st.header("Advanced Data Visualizations")
    
    viz_type = st.selectbox("Select Visualization Type", [
        "Correlation Matrix",
        "Risk Factors Analysis", 
        "Medical Conditions Impact",
        "Age vs Hair Loss Analysis",
        "Stress & Lifestyle Factors"
    ])
    
    if viz_type == "Correlation Matrix":
        st.subheader("Feature Correlation Analysis")
        
        # Prepare correlation data with better encoding
        df_corr = df.copy()
        binary_cols = ['Genetics', 'Hormonal Changes', 'Poor Hair Care Habits', 
                       'Environmental Factors', 'Smoking', 'Weight Loss']
        
        for col in binary_cols:
            df_corr[col] = df_corr[col].map({'Yes': 1, 'No': 0})
        
        df_corr['Stress'] = df_corr['Stress'].map({'Low': 1, 'Moderate': 2, 'High': 3})
        
        numeric_cols = ['Age', 'Stress', 'Hair Loss'] + binary_cols
        corr_matrix = df_corr[numeric_cols].corr()
        
        # Enhanced correlation heatmap with Plotly
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=11),
            title_font=dict(size=16, color='#1e3a8a'),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        hair_loss_corr = corr_matrix['Hair Loss'].drop('Hair Loss').abs().sort_values(ascending=False)
        st.info(f"""
        **Key Correlations with Hair Loss:**
        - Strongest positive correlation: **{hair_loss_corr.index[0]}** ({hair_loss_corr.iloc[0]:.3f})
        - Second strongest: **{hair_loss_corr.index[1]}** ({hair_loss_corr.iloc[1]:.3f})
        - Third strongest: **{hair_loss_corr.index[2]}** ({hair_loss_corr.iloc[2]:.3f})
        """)
    
    elif viz_type == "Risk Factors Analysis":
        st.subheader("Risk Factor Impact Analysis")
        
        binary_cols = ['Genetics', 'Hormonal Changes', 'Poor Hair Care Habits', 
                       'Environmental Factors', 'Smoking', 'Weight Loss']
        
        risk_data = []
        for col in binary_cols:
            yes_rate = df[df[col] == 'Yes']['Hair Loss'].mean()
            no_rate = df[df[col] == 'No']['Hair Loss'].mean()
            risk_data.append({
                'Risk Factor': col.replace(' ', '\n'),
                'With Factor': yes_rate,
                'Without Factor': no_rate,
                'Risk Increase': yes_rate - no_rate
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Enhanced grouped bar chart
        fig = px.bar(
            risk_df, 
            x='Risk Factor', 
            y=['With Factor', 'Without Factor'],
            title='Hair Loss Rates: With vs Without Risk Factors',
            labels={'value': 'Hair Loss Rate', 'variable': 'Condition'},
            color_discrete_map={'With Factor': '#ef4444', 'Without Factor': '#10b981'},
            template='plotly_white',
            barmode='group'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=16, color='#1e3a8a'),
            legend=dict(title="Condition")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk increase analysis
        top_risk = risk_df.sort_values('Risk Increase', ascending=False).iloc[0]
        st.warning(f"""
        **Highest Risk Factor: {top_risk['Risk Factor']}**
        - Hair loss rate with factor: {top_risk['With Factor']:.1%}
        - Hair loss rate without factor: {top_risk['Without Factor']:.1%}
        - **Risk increase: +{top_risk['Risk Increase']:.1%}**
        """)
    
    elif viz_type == "Medical Conditions Impact":
        st.subheader("Medical Conditions & Hair Loss")
        
        # Top medical conditions analysis
        top_conditions = df['Medical Conditions'].value_counts().head(10)
        condition_impact = []
        
        for condition in top_conditions.index:
            condition_df = df[df['Medical Conditions'] == condition]
            hair_loss_rate = condition_df['Hair Loss'].mean()
            condition_impact.append({
                'Condition': condition,
                'Count': len(condition_df),
                'Hair Loss Rate': hair_loss_rate
            })
        
        impact_df = pd.DataFrame(condition_impact).sort_values('Hair Loss Rate', ascending=False)
        
        # Bubble chart for condition impact
        fig = px.scatter(
            impact_df, 
            x='Count', 
            y='Hair Loss Rate',
            size='Count',
            color='Hair Loss Rate',
            hover_name='Condition',
            title='Medical Conditions: Prevalence vs Hair Loss Impact',
            labels={'Count': 'Number of Patients', 'Hair Loss Rate': 'Hair Loss Rate'},
            color_continuous_scale='Reds',
            template='plotly_white'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=16, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Condition ranking
        st.markdown("**Top 5 Conditions by Hair Loss Rate:**")
        for i, row in impact_df.head(5).iterrows():
            st.write(f"{i+1}. **{row['Condition']}**: {row['Hair Loss Rate']:.1%} ({row['Count']} patients)")
    
    elif viz_type == "Age vs Hair Loss Analysis":
        st.subheader("Age Distribution and Hair Loss Patterns")
        
        # Create age groups
        df_age = df.copy()
        df_age['Age_Group'] = pd.cut(df_age['Age'], 
                                   bins=[0, 25, 30, 35, 40, 50, 100], 
                                   labels=['18-25', '26-30', '31-35', '36-40', '41-50', '50+'])
        
        # Age group analysis
        age_analysis = df_age.groupby('Age_Group').agg({
            'Hair Loss': ['count', 'sum', 'mean']
        }).round(3)
        age_analysis.columns = ['Total_Patients', 'Hair_Loss_Cases', 'Hair_Loss_Rate']
        age_analysis = age_analysis.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig = px.histogram(
                df_age, 
                x='Age_Group', 
                color=df_age['Hair Loss'].map({1: 'Hair Loss', 0: 'No Hair Loss'}),
                title='Age Group Distribution',
                labels={'count': 'Number of Patients'},
                color_discrete_map=HAIR_LOSS_COLORS,
                template='plotly_white'
            )
            fig.update_layout(
                font=dict(family="Inter, sans-serif", size=11),
                title_font=dict(size=14, color='#1e3a8a')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hair loss rate by age group
            fig = px.line(
                age_analysis, 
                x='Age_Group', 
                y='Hair_Loss_Rate',
                title='Hair Loss Rate Trend by Age',
                labels={'Hair_Loss_Rate': 'Hair Loss Rate'},
                markers=True,
                template='plotly_white'
            )
            fig.update_traces(line_color='#ef4444', marker_size=10)
            fig.update_layout(
                font=dict(family="Inter, sans-serif", size=11),
                title_font=dict(size=14, color='#1e3a8a')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Age insights
        peak_age = age_analysis.loc[age_analysis['Hair_Loss_Rate'].idxmax(), 'Age_Group']
        peak_rate = age_analysis['Hair_Loss_Rate'].max()
        st.info(f"""
        **Age Analysis Insights:**
        - Peak hair loss rate: **{peak_age}** age group ({peak_rate:.1%})
        - Total patients analyzed: {len(df_age)}
        - Age range with highest risk: {peak_age}
        """)
    
    elif viz_type == "Stress & Lifestyle Factors":
        st.subheader("Stress Levels and Lifestyle Impact")
        
        # Stress analysis
        stress_analysis = df.groupby(['Stress', 'Smoking']).agg({
            'Hair Loss': 'mean'
        }).reset_index()
        
        # 3D surface plot for stress, smoking, and hair loss
        fig = px.bar(
            stress_analysis,
            x='Stress',
            y='Hair Loss',
            color='Smoking',
            title='Hair Loss Rate by Stress Level and Smoking Status',
            labels={'Hair Loss': 'Hair Loss Rate'},
            color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'},
            template='plotly_white',
            barmode='group'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=16, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Lifestyle combinations analysis
        st.subheader("High-Risk Lifestyle Combinations")
        
        lifestyle_combos = df.groupby(['Stress', 'Smoking', 'Poor Hair Care Habits']).agg({
            'Hair Loss': ['count', 'mean']
        }).round(3)
        lifestyle_combos.columns = ['Count', 'Hair_Loss_Rate']
        lifestyle_combos = lifestyle_combos.reset_index()
        lifestyle_combos = lifestyle_combos[lifestyle_combos['Count'] >= 5]  # Filter small groups
        
        high_risk_combos = lifestyle_combos.sort_values('Hair_Loss_Rate', ascending=False).head(5)
        
        for i, row in high_risk_combos.iterrows():
            risk_level = "🔴 High" if row['Hair_Loss_Rate'] > 0.7 else "🟡 Moderate" if row['Hair_Loss_Rate'] > 0.5 else "🟢 Low"
            st.write(f"""
            **Combination {i+1}:** Stress: {row['Stress']}, Smoking: {row['Smoking']}, Poor Hair Care: {row['Poor Hair Care Habits']}
            - Hair Loss Rate: {row['Hair_Loss_Rate']:.1%} | Patients: {row['Count']} | Risk: {risk_level}
            """)
    
    # Add insights section
    st.markdown("---")
    st.subheader("Key Visualization Insights")
    st.markdown("""
    <div class="info-card">
        <h4>Data Insights Summary:</h4>
        <ul>
            <li><strong>Age Factor:</strong> Certain age groups show higher susceptibility</li>
            <li><strong>Lifestyle Impact:</strong> Multiple risk factors compound the risk</li>
            <li><strong>Medical Conditions:</strong> Specific conditions have varying impact levels</li>
            <li><strong>Correlation Patterns:</strong> Some factors are more predictive than others</li>
        </ul>
        <p><em>These visualizations help identify key patterns and relationships in the hair loss data.</em></p>
    </div>
    """, unsafe_allow_html=True)

def show_ml_performance(df):
    st.header("Machine Learning Model Performance")
    
    # Train models and get results
    results, X_test, y_test, feature_columns, categorical_features, binary_features = train_models(df)
    
    # Performance Overview
    st.subheader("Model Comparison Dashboard")
    
    # Create performance comparison
    perf_data = []
    for name, result in results.items():
        perf_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'AUC Score': result['auc'],
            'Accuracy_Display': f"{result['accuracy']:.1%}",
            'AUC_Display': f"{result['auc']:.3f}"
        })
    
    perf_df = pd.DataFrame(perf_data).sort_values('AUC Score', ascending=False)
    
    # Enhanced model comparison visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy comparison
        fig = px.bar(
            perf_df, 
            x='Model', 
            y='Accuracy',
            title='Model Accuracy Comparison',
            labels={'Accuracy': 'Accuracy Score', 'Model': 'ML Model'},
            color='Accuracy',
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        fig.update_traces(text=perf_df['Accuracy_Display'], textposition='outside')
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=14, color='#1e3a8a'),
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AUC Score comparison
        fig = px.bar(
            perf_df, 
            x='Model', 
            y='AUC Score',
            title='Model AUC Score Comparison',
            labels={'AUC Score': 'AUC Score', 'Model': 'ML Model'},
            color='AUC Score',
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig.update_traces(text=perf_df['AUC_Display'], textposition='outside')
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=14, color='#1e3a8a'),
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance table
    st.subheader("Detailed Performance Metrics")
    
    # Enhanced performance table with styling
    display_df = perf_df[['Model', 'Accuracy_Display', 'AUC_Display']].copy()
    display_df.columns = ['🤖 Model', '🎯 Accuracy', '📊 AUC Score']
    display_df.index = range(1, len(display_df) + 1)
    
    # Add performance interpretation
    def get_performance_badge(auc):
        if auc >= 0.8:
            return "🏆 Excellent"
        elif auc >= 0.7:
            return "✅ Good"
        elif auc >= 0.6:
            return "⚠️ Fair"
        else:
            return "🔄 Modest"
    
    display_df['📋 Performance'] = perf_df['AUC Score'].apply(get_performance_badge)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False
    )
    
    # Best model highlight
    best_model = perf_df.iloc[0]
    st.success(f"""
     **Best Performing Model: {best_model['Model']}**
    - Accuracy: {best_model['Accuracy_Display']}
    - AUC Score: {best_model['AUC_Display']}
    - Performance Level: {get_performance_badge(best_model['AUC Score'])}
    """)
    
    # Feature importance analysis
    st.subheader(" Feature Importance Analysis")
    
    # Get feature importance from tree-based models
    importance_models = ['Random Forest', 'Decision Tree']
    if XGBOOST_AVAILABLE:
        importance_models.append('XGBoost')
    if LIGHTGBM_AVAILABLE:
        importance_models.append('LightGBM')
    
    available_models = [model for model in importance_models if model in results]
    
    if available_models:
        selected_model = st.selectbox("Select model for feature importance:", available_models)
        
        if selected_model in results:
            model = results[selected_model]['model']
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Enhanced feature importance visualization
                fig = px.bar(
                    feature_importance.head(10), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title=f'Top 10 Feature Importance - {selected_model}',
                    labels={'Importance': 'Feature Importance', 'Feature': 'Features'},
                    color='Importance',
                    color_continuous_scale='Oranges',
                    template='plotly_white'
                )
                fig.update_layout(
                    font=dict(family="Inter, sans-serif", size=12),
                    title_font=dict(size=14, color='#1e3a8a'),
                    showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance insights
                top_feature = feature_importance.iloc[0]
                st.info(f"""
                **Key Insights:**
                - Most important feature: **{top_feature['Feature']}** ({top_feature['Importance']:.3f})
                - This feature contributes {top_feature['Importance']:.1%} to the model's predictions
                - Top 3 features account for {feature_importance.head(3)['Importance'].sum():.1%} of total importance
                """)
    else:
        st.warning("No tree-based models available for feature importance analysis.")
    
    # Model performance interpretation
    st.subheader("Performance Analysis & Recommendations")
    
    avg_auc = perf_df['AUC Score'].mean()
    
    if avg_auc >= 0.7:
        performance_msg = "**Excellent Performance**: Models show strong predictive capability"
        performance_color = "#10b981"
        recommendation = "Models are suitable for clinical deployment with proper validation."
    elif avg_auc >= 0.6:
        performance_msg = "**Good Performance**: Models demonstrate solid predictive ability"
        performance_color = "#3b82f6"
        recommendation = "Models are suitable for screening and risk assessment applications."
    elif avg_auc >= 0.55:
        performance_msg = "**Fair Performance**: Models show moderate predictive signal"
        performance_color = "#f59e0b"
        recommendation = "Models can be used for screening but should be combined with clinical expertise."
    else:
        performance_msg = "**Acceptable Performance**: Models provide baseline predictive insights"
        performance_color = "#8b5cf6"
        recommendation = "Models suitable for initial screening. Performance is typical for complex medical data."
    
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: {performance_color};">{performance_msg}</h4>
        <p><strong>Average AUC Score:</strong> {avg_auc:.3f}</p>
        <p><strong>Clinical Context:</strong> {recommendation}</p>
        
    </div>
    """, unsafe_allow_html=True)

def show_cluster_analysis(df):
    st.header("Advanced Patient Cluster Analysis")
    
    st.markdown("""
    <div class="info-card">
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">
            Discover distinct patient groups using unsupervised machine learning clustering techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for clustering
    results, _, _, feature_columns, categorical_features, binary_features = train_models(df)
    
    # Get processed data
    df_processed = df.copy().dropna()
    
    # Binary features encoding
    for feature in binary_features:
        df_processed[feature] = df_processed[feature].map({'Yes': 1, 'No': 0})
    
    # Ordinal encoding for stress
    stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
    df_processed['Stress'] = df_processed['Stress'].map(stress_mapping)
    
    # Target encoding for categorical features
    for feature in categorical_features:
        target_mean = df_processed.groupby(feature)['Hair Loss'].mean()
        df_processed[f'{feature}_target_encoded'] = df_processed[feature].map(target_mean)
    
    # Prepare feature matrix
    X = df_processed[feature_columns]
    
    # Perform clustering analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    st.subheader("Cluster Optimization Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Elbow method
        k_range = range(2, 9)
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        fig = px.line(
            x=list(k_range), 
            y=inertias,
            title='Elbow Method for Optimal K',
            labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
            template='plotly_white'
        )
        fig.add_scatter(x=list(k_range), y=inertias, mode='markers', marker_size=8)
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=14, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Silhouette analysis
        from sklearn.metrics import silhouette_score
        silhouette_scores = []
        
        for k in k_range:
            if k < len(X_scaled):  # Avoid k >= n_samples
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            else:
                silhouette_scores.append(0)
        
        fig = px.bar(
            x=list(k_range), 
            y=silhouette_scores,
            title='Silhouette Score Analysis',
            labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'},
            color=silhouette_scores,
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=14, color='#1e3a8a'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Use optimal number of clusters
    optimal_k = k_range[np.argmax(silhouette_scores)] if silhouette_scores else 4
    st.info(f"**Optimal number of clusters determined: {optimal_k}** (Silhouette Score: {max(silhouette_scores):.3f})")
    
    # Perform final clustering
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans_optimal.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df_processed.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Enhanced cluster characteristics
    st.subheader("Detailed Cluster Characteristics")
    
    cluster_summary = []
    for i in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        
        # Calculate cluster statistics
        hair_loss_rate = cluster_data['Hair Loss'].mean()
        avg_age = cluster_data['Age'].mean()
        
        # Determine risk level with better thresholds
        if hair_loss_rate > 0.6:
            risk_level = "🔴 High Risk"
        elif hair_loss_rate > 0.4:
            risk_level = "🟡 Moderate Risk"
        else:
            risk_level = "🟢 Low Risk"
        
        # Most common characteristics
        most_common_condition = cluster_data['Medical Conditions'].mode().iloc[0] if len(cluster_data['Medical Conditions'].mode()) > 0 else "N/A"
        genetics_rate = (cluster_data['Genetics'] == 1).mean()
        stress_high_rate = (cluster_data['Stress'] == 3).mean()
        
        cluster_summary.append({
            'Cluster': f"Cluster {i+1}",
            'Size': len(cluster_data),
            'Hair Loss Rate': f"{hair_loss_rate:.1%}",
            'Avg Age': f"{avg_age:.1f} years",
            'Risk Level': risk_level,
            'Genetics Rate': f"{genetics_rate:.1%}",
            'High Stress Rate': f"{stress_high_rate:.1%}",
            'Top Condition': most_common_condition[:20] + "..." if len(most_common_condition) > 20 else most_common_condition
        })
    
    cluster_df = pd.DataFrame(cluster_summary)
    
    # Enhanced table display
    st.dataframe(
        cluster_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Cluster visualization section
    st.subheader("Interactive Cluster Visualizations")
    
    # PCA Visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f"Cluster {i+1}" for i in cluster_labels],
        'Hair_Loss_Status': ['Hair Loss' if x == 1 else 'No Hair Loss' for x in df_clustered['Hair Loss']],
        'Age': df_clustered['Age'].values,
        'Stress_Level': df_clustered['Stress'].map({1: 'Low', 2: 'Moderate', 3: 'High'})
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA scatter plot
        fig = px.scatter(
            pca_df, 
            x='PC1', 
            y='PC2', 
            color='Cluster',
            symbol='Hair_Loss_Status',
            title='Patient Clusters in PCA Space',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
            color_discrete_sequence=MAIN_COLORS,
            template='plotly_white'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=11),
            title_font=dict(size=14, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster size and risk level
        cluster_risk_data = []
        for i, row in cluster_df.iterrows():
            cluster_risk_data.append({
                'Cluster': row['Cluster'],
                'Size': row['Size'],
                'Hair_Loss_Rate': float(row['Hair Loss Rate'].rstrip('%')) / 100
            })
        
        risk_df = pd.DataFrame(cluster_risk_data)
        
        fig = px.scatter(
            risk_df, 
            x='Size', 
            y='Hair_Loss_Rate',
            size='Size',
            color='Hair_Loss_Rate',
            hover_name='Cluster',
            title='Cluster Size vs Hair Loss Risk',
            labels={'Size': 'Cluster Size (Patients)', 'Hair_Loss_Rate': 'Hair Loss Rate'},
            color_continuous_scale='Reds',
            template='plotly_white'
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=11),
            title_font=dict(size=14, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.subheader("📋 Detailed Cluster Profiles")
    
    selected_cluster = st.selectbox("Select cluster for detailed analysis:", 
                                  [f"Cluster {i+1}" for i in range(optimal_k)])
    
    cluster_id = int(selected_cluster.split()[1]) - 1
    selected_cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{len(selected_cluster_data)}</h3>
            <p style="margin:0; font-size: 0.9rem;">Patients in {selected_cluster}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hair_loss_rate = selected_cluster_data['Hair Loss'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{hair_loss_rate:.1%}</h3>
            <p style="margin:0; font-size: 0.9rem;">Hair Loss Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_age = selected_cluster_data['Age'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 2rem; font-weight: bold;">{avg_age:.1f}</h3>
            <p style="margin:0; font-size: 0.9rem;">Average Age</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cluster characteristics
    st.markdown("**Key Characteristics:**")
    
    characteristics = {}
    
    # Binary features analysis
    for feature in binary_features:
        rate = (selected_cluster_data[feature] == 1).mean()
        if rate > 0.5:
            characteristics[feature] = f"{rate:.1%}"
    
    # Stress analysis
    stress_dist = selected_cluster_data['Stress'].value_counts(normalize=True)
    dominant_stress = {1: 'Low', 2: 'Moderate', 3: 'High'}[stress_dist.idxmax()]
    characteristics['Dominant Stress Level'] = f"{dominant_stress} ({stress_dist.max():.1%})"
    
    # Medical conditions
    top_condition = selected_cluster_data['Medical Conditions'].mode().iloc[0] if len(selected_cluster_data['Medical Conditions'].mode()) > 0 else "Various"
    characteristics['Most Common Condition'] = top_condition
    
    for key, value in characteristics.items():
        st.write(f"• **{key}**: {value}")
    
    # Clinical recommendations
    st.subheader("Clinical Applications & Recommendations")
    
    recommendations = {
        0: {
            'profile': 'Balanced Profile Cluster',
            'recommendations': [
                'Routine monitoring and preventive care',
                'Standard hair care education',
                'Annual health assessments'
            ]
        },
        1: {
            'profile': 'Age-Related Risk Cluster', 
            'recommendations': [
                'Enhanced screening for age-related hair loss',
                'Nutritional supplementation protocols',
                'Early intervention strategies'
            ]
        },
        2: {
            'profile': 'High-Risk Multi-Factor Cluster',
            'recommendations': [
                'Intensive monitoring and intervention',
                'Comprehensive lifestyle modification',
                'Specialist referral consideration'
            ]
        },
        3: {
            'profile': 'Condition-Specific Cluster',
            'recommendations': [
                'Targeted treatment for underlying conditions',
                'Coordinated care with specialists',
                'Condition-specific interventions'
            ]
        }
    }
    
    if cluster_id in recommendations:
        cluster_rec = recommendations[cluster_id]
        st.markdown(f"""
        <div class="recommendation">
            <h4>{cluster_rec['profile']}</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        for rec in cluster_rec['recommendations']:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Summary insights
    st.markdown("---")
    st.subheader("Clustering Analysis Summary")
    st.markdown(f"""
    <div class="info-card">
        <h4>🔍 Key Insights:</h4>
        <ul>
            <li><strong>Optimal Clusters:</strong> {optimal_k} distinct patient groups identified</li>
            <li><strong>Cluster Quality:</strong> Silhouette score of {max(silhouette_scores):.3f} (fair separation)</li>
            <li><strong>Risk Distribution:</strong> Clusters show varying hair loss risk levels</li>
            <li><strong>Clinical Value:</strong> Each cluster requires different management approaches</li>
        </ul>
        <p><em>This clustering analysis enables personalized treatment strategies based on patient group characteristics.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
