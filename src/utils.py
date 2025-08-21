"""
Shared utility functions for Asthma Disease Prediction Model.
Professional utilities following ThriveAfrica patterns for ML consistency.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Union, List
from datetime import datetime
import hashlib
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_asthma_input(input_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate input data for asthma severity prediction.
    
    Args:
        input_data: Dictionary containing patient data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 
                      'DietQuality', 'SleepQuality', 'PollutionExposure']
    
    try:
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Validate data types and ranges
        validations = [
            ('Age', lambda x: isinstance(x, (int, float)) and 0 <= x <= 120),
            ('BMI', lambda x: isinstance(x, (int, float)) and 10 <= x <= 50),
            ('Smoking', lambda x: x in [0, 1]),  # Binary
            ('PhysicalActivity', lambda x: isinstance(x, (int, float)) and 0 <= x <= 10),
            ('DietQuality', lambda x: isinstance(x, (int, float)) and 0 <= x <= 10),
            ('SleepQuality', lambda x: isinstance(x, (int, float)) and 0 <= x <= 10),
            ('PollutionExposure', lambda x: isinstance(x, (int, float)) and 0 <= x <= 10),
        ]
        
        for field, validator in validations:
            if field in input_data and not validator(input_data[field]):
                return False, f"Invalid value for {field}: {input_data[field]}"
        
        return True, "Valid input data"
        
    except Exception as e:
        logger.error(f"Error validating asthma input data: {str(e)}")
        return False, f"Validation error: {str(e)}"


def calculate_risk_factors(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate additional risk factors and health metrics.
    
    Args:
        patient_data: Dictionary with patient information
        
    Returns:
        Dictionary with calculated risk factors
    """
    try:
        age = patient_data.get('Age', 0)
        bmi = patient_data.get('BMI', 0)
        smoking = patient_data.get('Smoking', 0)
        physical_activity = patient_data.get('PhysicalActivity', 0)
        pollution_exposure = patient_data.get('PollutionExposure', 0)
        
        # Age risk classification
        if age < 18:
            age_risk = "Child/Adolescent Risk"
        elif age < 65:
            age_risk = "Adult"
        else:
            age_risk = "Senior Risk"
        
        # BMI classification
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        # Lifestyle risk score (0-100)
        lifestyle_risk = (
            (smoking * 30) +  # Smoking contributes heavily
            (max(0, 10 - physical_activity) * 4) +  # Low activity increases risk
            (pollution_exposure * 3)  # Environmental factors
        )
        lifestyle_risk = min(lifestyle_risk, 100)
        
        # Overall risk category
        if lifestyle_risk < 30:
            risk_category = "Low Risk"
        elif lifestyle_risk < 60:
            risk_category = "Moderate Risk"
        else:
            risk_category = "High Risk"
        
        return {
            'age_risk_group': age_risk,
            'bmi_category': bmi_category,
            'lifestyle_risk_score': round(lifestyle_risk, 1),
            'risk_category': risk_category,
            'smoking_status': "Smoker" if smoking else "Non-smoker"
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk factors: {str(e)}")
        return {
            'age_risk_group': "Unknown",
            'bmi_category': "Unknown", 
            'lifestyle_risk_score': 0.0,
            'risk_category': "Unknown",
            'smoking_status': "Unknown"
        }


def interpret_severity_prediction(prediction: int, probability: float) -> Dict[str, str]:
    """
    Interpret the asthma severity prediction for user understanding.
    
    Args:
        prediction: Model prediction (0=Mild, 1=Moderate, 2=Severe)
        probability: Prediction confidence
        
    Returns:
        Dictionary with interpretation and recommendations
    """
    severity_map = {
        0: "Mild Asthma",
        1: "Moderate Asthma", 
        2: "Severe Asthma"
    }
    
    severity = severity_map.get(prediction, "Unknown")
    
    recommendations = {
        0: [
            "Continue regular exercise and healthy lifestyle",
            "Monitor symptoms and maintain medication as prescribed",
            "Avoid known triggers and maintain good air quality",
            "Regular check-ups with healthcare provider"
        ],
        1: [
            "Increased monitoring of symptoms and peak flow",
            "Review medication plan with healthcare provider",
            "Consider environmental modifications to reduce triggers",
            "Maintain consistent sleep schedule and stress management"
        ],
        2: [
            "Immediate consultation with specialist required",
            "Strict adherence to medication regimen essential",
            "Environmental control measures critical",
            "Consider action plan for severe symptom management"
        ]
    }
    
    confidence_level = "High" if probability > 0.8 else "Moderate" if probability > 0.6 else "Low"
    
    return {
        'severity_level': severity,
        'confidence': confidence_level,
        'recommendations': recommendations.get(prediction, ["Consult healthcare provider"]),
        'confidence_score': f"{probability:.1%}"
    }