# EHR Nexus ML - Dataset Schema Guide

## Overview

To train the ML models, you need a labeled dataset. Below are the recommended CSV formats for each task type. You can also use JSON or JSONL.

---

## 1. Symptom Correlation Dataset

**Purpose:** Train the model to recognize which symptoms are equivalent or clinically correlated.

**Recommended file name:** `symptom_pairs.csv`

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `symptom_a` | text | Yes | First symptom name | `"headache"` |
| `symptom_b` | text | Yes | Second symptom name | `"cephalgia"` |
| `label` | int (0/1) | Yes | 1 = same/correlated, 0 = different | `1` |
| `severity_a` | text | No | Severity of symptom_a | `"mild"` |
| `severity_b` | text | No | Severity of symptom_b | `"moderate"` |
| `body_location` | text | No | Body location | `"head"` |

**Example rows:**
```csv
symptom_a,symptom_b,label,severity_a,severity_b
headache,cephalgia,1,mild,moderate
fever,pyrexia,1,severe,severe
fever,cough,0,moderate,mild
chest pain,angina,1,severe,mild
headache,fever,0,mild,moderate
nausea,vomiting,1,mild,severe
shortness of breath,dyspnea,1,severe,severe
abdominal pain,stomach ache,1,moderate,mild
```

**Total recommended:** 500+ rows minimum, 2000+ for good results.

---

## 2. Diagnosis Prediction Dataset

**Purpose:** Train the model to predict diagnoses from symptoms, patient demographics, and lab values.

**Recommended file name:** `diagnosis_data.csv`

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `patient_id` | text | No | Patient identifier | `"P001"` |
| `age` | int | No | Patient age | `65` |
| `sex` | text | No | Patient sex | `"Male"` |
| `symptom_names` | text | Yes | Comma-separated symptoms | `"fever,cough,dyspnea"` |
| `confirmed_diagnosis` | text | Yes | **LABEL** - the correct diagnosis | `"Pneumonia"` |
| `diagnosis_code` | text | No | ICD-10 code | `"J15.9"` |
| `lab_parameters` | text | No | Comma-separated lab params | `"WBC,CRP"` |
| `lab_values` | text | No | Comma-separated lab values | `"15.2,85"` |
| `lab_units` | text | No | Comma-separated units | `"K/uL,mg/L"` |

**Example rows:**
```csv
patient_id,age,sex,symptom_names,confirmed_diagnosis,diagnosis_code,lab_parameters,lab_values
P001,72,Male,"fever,cough,dyspnea",Pneumonia,J15.9,"WBC,CRP","15.2,85"
P002,45,Female,"fatigue,weight_loss,polydipsia",Diabetes,E11.9,"Glucose,HbA1c","280,9.8"
P003,58,Male,"chest_pain,diaphoresis",Myocardial Infarction,I21.9,"Troponin,CK-MB","4.5,25"
P004,34,Female,"headache,blurred_vision",Hypertension,I10,"None","None"
P005,25,Male,"sore_throat,cough,fever",Upper Respiratory Infection,J06.9,"WBC","8.1"
```

**Total recommended:** 1000+ rows minimum, 5000+ for good results.

---

## 3. Lab Anomaly Detection Dataset

**Purpose:** Train the model to detect abnormal lab values (unsupervised - no labels needed).

**Recommended file name:** `lab_data.csv`

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `patient_id` | text | No | Patient identifier | `"P001"` |
| `age` | int | No | Patient age | `65` |
| `sex` | text | No | Patient sex | `"Male"` |
| `parameter_name` | text | Yes | Lab test name | `"Hemoglobin"` |
| `value` | float | Yes | Lab value | `14.2` |
| `unit` | text | No | Measurement unit | `"g/dL"` |
| `ref_range_low` | float | No | Reference range low | `13.0` |
| `ref_range_high` | float | No | Reference range high | `17.0` |
| `is_abnormal` | int (0/1) | Optional | **LABEL** - for evaluation | `0` |
| `diagnosis_code` | text | No | Associated diagnosis | `"E11.9"` |

**Example rows:**
```csv
patient_id,age,sex,parameter_name,value,unit,ref_range_low,ref_range_high,is_abnormal
P001,65,Male,Hemoglobin,14.2,g/dL,13.0,17.0,0
P001,65,Male,Glucose,280,mg/dL,70,110,1
P002,45,Female,White Blood Cell,15.2,K/uL,4.0,11.0,1
P003,58,Male,Potassium,3.8,mmol/L,3.5,5.0,0
P004,34,Female,TSH,8.5,mIU/L,0.4,4.0,1
P005,25,Male,CRP,2.0,mg/L,0,5,0
```

**Total recommended:** 500+ rows minimum, 2000+ for good results.

---

## 4. Full Patient Record Dataset (For Correlation Engine Enhancement)

**Purpose:** End-to-end training that correlates symptoms, diagnoses, and labs together.

**Recommended file name:** `patient_records.csv`

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `consultation_id` | text | Yes | Unique visit ID | `"C001"` |
| `patient_id` | text | Yes | Patient identifier | `"P001"` |
| `patient_age` | int | No | Age at visit | `65` |
| `patient_sex` | text | No | Sex | `"Male"` |
| `consultation_date` | date | No | Visit date | `"2025-06-01"` |
| `symptom_names` | text | Yes | All symptoms (comma-sep) | `"fever,cough,dyspnea"` |
| `severity_list` | text | No | Severities (comma-sep) | `"high,moderate,severe"` |
| `diagnosis_names` | text | Yes | All diagnoses (comma-sep) | `"Pneumonia,Hypertension"` |
| `diagnosis_codes` | text | No | ICD-10 codes (comma-sep) | `"J15.9,I10"` |
| `lab_parameters` | text | No | Lab names (comma-sep) | `"WBC,CRP,Hemoglobin"` |
| `lab_values` | text | No | Lab values (comma-sep) | `"15.2,85,14.2"` |
| `doctor_notes` | text | No | Free text notes | `"Patient presents with..."` |

**Example rows:**
```csv
consultation_id,patient_id,patient_age,patient_sex,symptom_names,diagnosis_names,diagnosis_codes,lab_parameters,lab_values
C001,P001,72,Male,"fever,cough,dyspnea","Pneumonia,Hypertension","J15.9,I10","WBC,CRP","15.2,85"
C002,P002,45,Female,"fatigue,weight_loss,polydipsia","Diabetes Mellitus,E11.9","E11.9","Glucose,HbA1c","280,9.8"
C003,P003,58,Male,"chest_pain,diaphoresis","Myocardial Infarction","I21.9","Troponin,CK-MB","4.5,25"
```

**Total recommended:** 2000+ rows minimum, 10000+ for good results.

---

## Quick Start

1. **Place your CSV file** in the `ehr_nexus_ai/data/` directory
2. **Preview it** to check the format:
   ```python
   from ehr_nexus_ai.ml import ModelTrainer
   trainer = ModelTrainer()
   trainer.preview("ehr_nexus_ai/data/your_dataset.csv")
   ```
3. **Train the model**:
   ```python
   results = trainer.train("ehr_nexus_ai/data/your_dataset.csv", task="auto")
   print(f"Accuracy: {results['test_metrics']['accuracy']:.2%}")
   ```

> **Important:** The dataset MUST have a label column. For symptom correlation, use `label` (0 or 1). For diagnosis prediction, use `confirmed_diagnosis`. For lab anomaly, use `is_abnormal`. If using `task="auto"`, the system will try to find these columns automatically.