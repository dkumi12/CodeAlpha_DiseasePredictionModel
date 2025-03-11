# Asthma Severity Prediction Model

## Overview
This repository contains a machine learning model to predict asthma severity based on patient characteristics and environmental factors. The model was trained using a comprehensive dataset that was enhanced with additional features to improve predictive accuracy.

## Dataset
The model was trained using two versions of asthma-related datasets:
1. `AsthmaDiseasePrediction.csv` - Original dataset containing essential patient features. link (https://drive.google.com/file/d/1cYzT1k3EyIrzPUiOfrWRceBhtIzVEiRB/view?usp=drive_link)
2. `Enhanced_AsthmaDiseasePrediction.csv` - Enhanced dataset with additional features such as: link (https://drive.google.com/file/d/14z4d9nIOFKt-lT_vfglo444EXnljjW9V/view?usp=drive_link)
   - Pollen count
   - Air quality index
   - Exercise habits
   - Family history of respiratory diseases
   - Previous medication adherence

## Features
The model considers the following key features:
- Demographics: Age, Gender, Smoking status, BMI
- Medical history: Respiratory infection history, Allergy status
- Environmental factors: Pollution exposure, Pollen count, Air quality index
- Physiological measurements: Peak expiratory flow rate, Forced expiratory volume (FEV1)
- Lifestyle: Exercise habits, Previous medication adherence
- Family history: Family history of respiratory diseases

## Model Details
- Algorithm: Random Forest Classifier
- Training accuracy: 92%
- Cross-validation score: 88%
- Evaluation metrics:
  - Precision: 0.91
  - Recall: 0.90
  - F1-score: 0.90

## Usage
To use the model:
1. Ensure all requirements are installed.
2. Run the prediction script:
   ```bash
   python asthmapred.py
   ```
3. For web interface:
   ```bash
   python asthma_app.py
   ```

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- Flask

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, contact David Osei Kumi at 12dkumi@gmail.com
