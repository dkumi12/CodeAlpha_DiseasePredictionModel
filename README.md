# 🫁 Asthma Severity Prediction Model - Professional ML Healthcare Platform

[![CI/CD Pipeline](https://github.com/dkumi12/CodeAlpha_DiseasePredictionModel/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dkumi12/CodeAlpha_DiseasePredictionModel/actions/workflows/ci-cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)

> **Advanced machine learning platform for predicting asthma severity with comprehensive risk factor analysis, patient safety validation, and clinical decision support**

## 🎯 Medical AI Innovation

### **Clinical Decision Support**
- **🏥 Asthma Severity Classification**: ML-powered prediction of mild, moderate, or severe asthma
- **📊 Risk Factor Analysis**: Comprehensive assessment of lifestyle and environmental factors
- **⚕️ Clinical Recommendations**: Evidence-based suggestions for patient care
- **📈 Health Metrics**: BMI classification, lifestyle risk scoring, and age-based analysis

### **Professional Healthcare Engineering**
- **✅ Medical Data Validation**: Comprehensive input validation for patient safety
- **✅ Risk Assessment**: Multi-factor analysis including smoking, activity, pollution
- **✅ Interpretable AI**: Clear explanations of predictions for clinical use
- **✅ Safety-First Design**: Extensive error handling and edge case management
- **✅ Comprehensive Testing**: 8 comprehensive tests covering all scenarios
- **✅ CI/CD Pipeline**: Automated testing and model validation

### **Research & Academic Excellence**
- **📚 Enhanced Dataset**: Extended original dataset with environmental and lifestyle factors
- **🔬 Feature Engineering**: Pollen count, air quality, exercise habits, family history
- **📊 Model Performance**: Validated predictions with clinical relevance
- **📝 Documentation**: Complete research methodology and findings

## 🛠️ Technology Stack

### **Machine Learning**
- **Python 3.9+** - Modern Python with latest ML libraries
- **scikit-learn** - Professional ML model implementation
- **Pandas & NumPy** - Advanced data processing and analysis
- **Joblib** - Optimized model persistence and loading

### **Web Application**
- **Streamlit** - Interactive medical application interface
- **Professional UI** - Clinical-grade interface design
- **Real-time Predictions** - Instant asthma severity assessment
- **Risk Visualization** - Clear presentation of health factors

### **Development & Quality**
- **Pytest** - Comprehensive medical AI testing framework
- **GitHub Actions** - CI/CD with model validation and security scanning
- **Code Quality** - Medical-grade error handling and validation
- **Documentation** - Clinical and technical documentation

## 📊 Dataset & Features

### **Original Features**
- **Patient Demographics**: Age, Gender, BMI
- **Lifestyle Factors**: Smoking status, Physical Activity level
- **Health Indicators**: Diet Quality, Sleep Quality
- **Environmental**: Pollution Exposure levels

### **Enhanced Features** (Research Extension)
- **Environmental Data**: Pollen count, Air Quality Index (AQI)
- **Exercise Patterns**: Detailed physical activity habits and intensity
- **Medical History**: Family history of respiratory diseases
- **Treatment Adherence**: Previous medication compliance patterns
- **Seasonal Factors**: Weather patterns and seasonal trigger analysis

### **Target Variable**
- **Asthma Severity**: Three-class classification
  - **0**: Mild Asthma (manageable with basic treatment)
  - **1**: Moderate Asthma (requires consistent medication)
  - **2**: Severe Asthma (needs specialist care and intensive management)

## 🚀 Quick Start

### **Prerequisites**
- Python 3.9 or higher
- Git

### **Installation**
```bash
# Clone the repository
git clone https://github.com/dkumi12/CodeAlpha_DiseasePredictionModel.git
cd CodeAlpha_DiseasePredictionModel

# Create virtual environment (recommended for medical applications)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install joblib scikit-learn pandas numpy streamlit

# Run the medical application
streamlit run asthma_app.py
```

### **Using the Prediction System**
1. **Access Interface**: Open http://localhost:8501 in your browser
2. **Input Patient Data**: Enter demographic and health information
3. **Environmental Factors**: Add pollution exposure and lifestyle data
4. **Get Assessment**: Receive severity prediction with confidence score
5. **Review Recommendations**: View clinical suggestions and risk factors

## 📁 Project Structure

```
CodeAlpha_DiseasePredictionModel/
├── asthma_app.py                # Main Streamlit medical application
├── asthmapred.py               # Model training and evaluation script
├── asthma_severity_model.joblib # Trained ML model for predictions
├── src/                        # Professional utility modules
│   └── utils.py               # Medical validation and risk analysis
├── tests/                      # Comprehensive test suite
│   ├── test_utils.py          # Medical utility tests (8 tests)
│   └── __init__.py            # Test package initialization
├── Dataset/                    # Medical dataset storage
│   ├── Original/              # Base asthma dataset
│   └── Enhanced/              # Research-extended dataset
├── .github/workflows/          # CI/CD with medical validation
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore rules
└── README.md                  # This medical documentation
```

## 🧪 Testing & Medical Validation

### **Test Suite Results**
```bash
# Run comprehensive medical AI test suite
pytest tests/ -v

# Expected output:
tests/test_utils.py::test_validate_asthma_input_valid PASSED
tests/test_utils.py::test_validate_asthma_input_invalid_age PASSED  
tests/test_utils.py::test_validate_asthma_input_missing_fields PASSED
tests/test_utils.py::test_calculate_risk_factors PASSED
tests/test_utils.py::test_calculate_risk_factors_healthy_profile PASSED
tests/test_utils.py::test_interpret_severity_prediction PASSED
tests/test_utils.py::test_interpret_severity_prediction_severe PASSED
tests/test_utils.py::test_interpret_severity_prediction_edge_cases PASSED

8 passed in 2.90s (100% success rate)
```

### **Medical Validation Features**
- **Patient Safety**: Input validation preventing dangerous values
- **Clinical Ranges**: Age (0-120), BMI (10-50), activity levels (0-10)
- **Risk Assessment**: Multi-factor lifestyle and environmental analysis
- **Interpretability**: Clear explanations for clinical decision support

## 🏥 Clinical Applications

### **Healthcare Provider Use**
- **Screening Tool**: Initial asthma severity assessment for new patients
- **Risk Stratification**: Identify high-risk patients requiring specialist referral
- **Treatment Planning**: Inform medication and lifestyle intervention decisions
- **Progress Monitoring**: Track changes in severity over time

### **Research Applications**
- **Population Health**: Analyze asthma patterns across demographics
- **Environmental Studies**: Correlation between pollution and asthma severity
- **Intervention Effectiveness**: Measure impact of lifestyle modifications
- **Predictive Modeling**: Advance understanding of asthma risk factors

### **Patient Education**
- **Risk Awareness**: Help patients understand modifiable risk factors
- **Lifestyle Guidance**: Show impact of smoking, exercise, and diet choices
- **Severity Understanding**: Clear explanation of asthma severity levels
- **Treatment Motivation**: Visual representation of improvement potential

## 📈 Model Performance & Validation

### **Performance Metrics**
- **Training Methodology**: Cross-validation with medical dataset
- **Feature Importance**: Environmental factors and lifestyle choices ranked
- **Clinical Validation**: Predictions aligned with medical guidelines
- **Bias Testing**: Validated across demographic groups for fairness

### **Risk Factor Analysis**
- **Smoking Impact**: 30-point contribution to lifestyle risk score
- **Physical Activity**: Inverse relationship with asthma severity
- **Environmental Factors**: Pollution exposure correlation
- **BMI Relationship**: Weight status impact on respiratory health

## 🔒 Medical Ethics & Privacy

### **Patient Privacy**
- **No Data Storage**: Patient information processed locally, never stored
- **HIPAA Considerations**: Designed with healthcare privacy principles
- **Secure Processing**: All calculations performed locally
- **Audit Logging**: Basic prediction tracking without personal information

### **Medical Disclaimers**
- **Clinical Support Only**: Tool assists healthcare decisions, doesn't replace doctors
- **Professional Consultation**: Always recommend medical professional evaluation
- **Research Purpose**: Academic and educational applications emphasized
- **Validation Required**: Clinical validation needed before medical deployment

## 🚀 Future Enhancements

### **Planned Features**
- **Longitudinal Tracking**: Patient progress monitoring over time
- **Integration APIs**: EHR system integration capabilities
- **Mobile Application**: Patient-facing mobile app development
- **Advanced Analytics**: Population health insights and reporting

### **Research Directions**
- **Multi-Modal Data**: Integration with wearable device data
- **Federated Learning**: Privacy-preserving collaborative model training
- **Explainable AI**: Advanced interpretability for clinical decisions
- **Real-World Evidence**: Integration with actual patient outcomes

## 🤝 Contributing

Help advance healthcare AI and respiratory medicine research!

### **How to Contribute**
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/medical-enhancement`)
3. **Add comprehensive tests** for medical applications
4. **Follow medical ethics** and privacy guidelines
5. **Update documentation** with clinical validation
6. **Submit a pull request** with detailed medical justification

### **Development Standards**
- **Medical Grade**: Extra attention to validation and error handling
- **Privacy First**: No patient data storage or logging
- **Clinical Accuracy**: Evidence-based recommendations and interpretations
- **Professional Quality**: Healthcare-appropriate code and documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CodeAlpha** - Educational platform supporting AI development
- **Medical Research Community** - Asthma research and clinical guidelines
- **scikit-learn Team** - Machine learning framework excellence
- **Streamlit** - Enabling accessible medical AI applications
- **Healthcare Professionals** - Clinical insights and validation guidance

## ⚠️ Important Medical Disclaimer

**This tool is for educational and research purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment of asthma or any medical condition.**

---

<div align="center">

**Advancing respiratory health through responsible AI innovation** 🫁🤖

[![GitHub stars](https://img.shields.io/github/stars/dkumi12/CodeAlpha_DiseasePredictionModel.svg?style=social&label=Star)](https://github.com/dkumi12/CodeAlpha_DiseasePredictionModel)

</div>