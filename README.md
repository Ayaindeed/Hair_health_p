# Hair Fall Prediction Analysis

A comprehensive machine learning analysis of hair loss patterns and risk factors using a dataset of 999 patients.

## Dataset

- **Source**: [Predict Hair Fall.csv](https://www.kaggle.com/datasets/amitvkulkarni/hair-health/data)
- **Records**: 999 patients
- **Features**: 12 key factors (Age, Genetics, Medical Conditions, Nutritional Deficiencies, Stress, etc.)
- **Target**: Binary hair loss prediction (0/1)
- **Balance**: 49.7% hair loss rate (balanced dataset)

## Analysis Overview

### Level 1: Descriptive Statistics
- Average patient age: 34.2 years (range 18-49)
- Most common medical conditions: Dermatosis, Eczema, Psoriasis
- Top nutritional deficiencies: Biotin, Vitamin D, Magnesium
- Complete demographic and clinical profiling

### Level 2: Data Visualizations
- Age-group hair loss distribution analysis
- Correlation matrix of risk factors
- Medical condition impact visualizations
- Stress level and nutritional deficiency patterns
- Interactive Plotly charts for multi-dimensional analysis

### Level 3: Machine Learning
- **Logistic Regression**: 56.2% accuracy, 0.591 AUC (Best performing)
- **XGBoost**: 53.7% accuracy, 0.552 AUC
- **Random Forest**: 54.3% accuracy, 0.548 AUC
- **LightGBM**: 51.2% accuracy, 0.511 AUC
- **Cluster Analysis**: 4 distinct patient groups (silhouette score: 0.097)

## Key Findings

1. **Age is the strongest predictor** (feature importance: 0.205)
2. **Nutritional deficiencies** are the second most critical factor
3. **Medical conditions and treatments** significantly impact outcomes
4. **Four distinct patient clusters** with different risk profiles identified
5. **Model performance is modest (~56-59% accuracy)** indicating moderate predictive signal
6. **Age group 26-35** shows highest hair loss rate (53.6%)

## Clinical Applications

### Risk Stratification
- **Cluster 0**: Low risk (balanced profile)
- **Cluster 1**: Moderate risk (age-related)
- **Cluster 2**: High risk (multiple factors)
- **Cluster 3**: Variable risk (condition-specific)

### Evidence-Based Recommendations
- Early screening for ages 26-35 (highest risk period)
- Nutritional assessment and supplementation protocols
- Stress management interventions
- Medical evaluation for underlying conditions

## Streamlit Web Application

This project includes an interactive Streamlit web application for real-time analysis and predictions.

### Features
- **Dataset Overview**: Interactive exploration of patient demographics
- **Risk Assessment Tool**: Personal hair loss risk calculator
- **Data Visualizations**: Dynamic charts and correlation analysis
- **ML Model Performance**: Model comparison and feature importance
- **Cluster Analysis**: Patient segmentation with PCA visualization

### Running the Streamlit App

#### Windows
```bash
# Double-click or run:
run_streamlit.bat
```

#### Linux/Mac
```bash
# Make executable and run:
chmod +x run_streamlit.sh
./run_streamlit.sh
```

#### Manual Setup
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Results Summary

The analysis provides a robust foundation for:
- **Risk assessment**: Identify high-risk patients early
- **Personalized treatment**: Tailored interventions based on cluster profiles  
- **Clinical decision support**: Evidence-based prevention strategies
- **Healthcare resource allocation**: Data-driven patient prioritization

## Model Performance & Limitations

### Performance Analysis
- **Best Model**: Logistic Regression with 56.2% accuracy and 0.591 AUC
- **Performance Level**: Fair/Modest - significantly better than random (50%) but indicates moderate predictive signal
- **Clinical Relevance**: Suitable for screening and risk assessment when combined with clinical judgment

### Key Limitations
- **Modest accuracy (~56-59%)** suggests need for additional features (genetic data, biomarkers)
- Cross-sectional design limits causal inference
- External validation needed on different populations
- Longitudinal follow-up required for treatment outcomes
- Performance requires clinical validation before deployment

### Future Improvements
- Integration with genetic testing and biomarker data
- Larger, more diverse patient populations
- Advanced feature engineering and ensemble methods
- Clinical validation studies


---

## License

This project is for educational and research purposes.
