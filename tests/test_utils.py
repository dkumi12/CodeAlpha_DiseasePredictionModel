"""
Test suite for Asthma Disease Prediction utilities
"""
import pytest
import numpy as np
from src.utils import validate_asthma_input, calculate_risk_factors, interpret_severity_prediction

def test_validate_asthma_input_valid():
    """Test validation with valid asthma patient data"""
    valid_data = {
        'Age': 35,
        'Gender': 1,  # Assuming encoded
        'BMI': 25.0,
        'Smoking': 0,
        'PhysicalActivity': 7,
        'DietQuality': 8,
        'SleepQuality': 7,
        'PollutionExposure': 3
    }
    
    is_valid, message = validate_asthma_input(valid_data)
    assert is_valid == True
    assert "Valid input data" in message

def test_validate_asthma_input_invalid_age():
    """Test validation with invalid age"""
    invalid_data = {
        'Age': 150,  # Invalid age
        'Gender': 1,
        'BMI': 25.0,
        'Smoking': 0,
        'PhysicalActivity': 7,
        'DietQuality': 8,
        'SleepQuality': 7,
        'PollutionExposure': 3
    }
    
    is_valid, message = validate_asthma_input(invalid_data)
    assert is_valid == False
    assert "Invalid value for Age" in message

def test_validate_asthma_input_missing_fields():
    """Test validation with missing required fields"""
    incomplete_data = {
        'Age': 35,
        'BMI': 25.0
        # Missing other required fields
    }
    
    is_valid, message = validate_asthma_input(incomplete_data)
    assert is_valid == False
    assert "Missing required fields" in message

def test_calculate_risk_factors():
    """Test risk factor calculations"""
    patient_data = {
        'Age': 45,
        'BMI': 28.0,
        'Smoking': 1,
        'PhysicalActivity': 3,
        'PollutionExposure': 7
    }
    
    risk_factors = calculate_risk_factors(patient_data)
    
    assert risk_factors['age_risk_group'] == "Adult"
    assert risk_factors['bmi_category'] == "Overweight"
    assert risk_factors['smoking_status'] == "Smoker"
    assert risk_factors['lifestyle_risk_score'] > 50  # High risk due to smoking and low activity
    assert risk_factors['risk_category'] in ["Low Risk", "Moderate Risk", "High Risk"]

def test_calculate_risk_factors_healthy_profile():
    """Test risk calculations for healthy patient"""
    healthy_patient = {
        'Age': 30,
        'BMI': 22.0,
        'Smoking': 0,
        'PhysicalActivity': 8,
        'PollutionExposure': 2
    }
    
    risk_factors = calculate_risk_factors(healthy_patient)
    
    assert risk_factors['age_risk_group'] == "Adult"
    assert risk_factors['bmi_category'] == "Normal"
    assert risk_factors['smoking_status'] == "Non-smoker"
    assert risk_factors['lifestyle_risk_score'] < 30  # Should be low risk
    assert risk_factors['risk_category'] == "Low Risk"

def test_interpret_severity_prediction():
    """Test severity prediction interpretation"""
    # Test mild asthma prediction
    interpretation = interpret_severity_prediction(0, 0.85)
    
    assert interpretation['severity_level'] == "Mild Asthma"
    assert interpretation['confidence'] == "High"
    assert len(interpretation['recommendations']) >= 3
    assert "Continue regular exercise" in interpretation['recommendations'][0]

def test_interpret_severity_prediction_severe():
    """Test severe asthma prediction interpretation"""
    interpretation = interpret_severity_prediction(2, 0.92)
    
    assert interpretation['severity_level'] == "Severe Asthma"
    assert interpretation['confidence'] == "High"
    assert len(interpretation['recommendations']) >= 3
    assert "specialist" in interpretation['recommendations'][0].lower()

def test_interpret_severity_prediction_edge_cases():
    """Test interpretation with edge cases"""
    # Test with low confidence
    interpretation = interpret_severity_prediction(1, 0.55)
    
    assert interpretation['severity_level'] == "Moderate Asthma"
    assert interpretation['confidence'] == "Low"
    assert interpretation['confidence_score'] == "55.0%"