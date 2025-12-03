# Student Success Prediction: A Two-Stage Machine Learning Approach

Machine learning system predicting student outcomes (Dropout, Enrolled, Graduate) with **76.72% accuracy**, enabling early intervention through a novel two-stage framework.

## Project Overview

This project develops a comprehensive machine learning approach to predict student success using data from 4,424 students at a Portuguese higher education institution. The analysis reveals a surprising finding: **structural factors present at admission contribute 53.2% of predictive power**, while academic performance contributes 46.8%. This challenges conventional assumptions that academic factors dominate student outcomes.

### Key Results

- **Full Model (XGBoost)**: 76.72% test accuracy, identifying 76% of dropout cases
- **Background-Only Model**: 64.18% accuracy using only admission data, catching 59% of dropouts before classes start
- **Two-Stage Intervention System**: Screen at admission (Stage 1) → Monitor after grades (Stage 2)
- **Actionable Insights**: Financial stress, program mismatch, and demographic factors are primary dropout predictors

Early identification of at-risk students enables universities to:
- Implement proactive financial aid and support before classes begin
- Deploy targeted academic interventions based on semester performance
- Address structural barriers (financial, demographic) and academic challenges
- Improve retention rates through data-driven resource allocation

## Dataset

**Source:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) from UCI Machine Learning Repository

The dataset contains information from a Portuguese higher education institution with:
- **4,424 student records** across multiple undergraduate programs
- **36 features** including demographics, socioeconomic factors, academic path, and performance metrics
- **3 target classes:** Dropout, Enrolled, Graduate
- **No missing values** - data has been rigorously preprocessed
- **Class imbalance** - realistic representation of student outcomes

## Classification Task

**Target Variable:** Student outcome at the end of normal course duration
- **Dropout** - student left before completing degree
- **Enrolled** - student still actively pursuing degree
- **Graduate** - student successfully completed degree

**Features Include:**
- Demographics (age, gender, marital status)
- Socioeconomic factors (parents' education, occupation)
- Academic background (previous qualifications, admission grades)
- Enrollment details (application mode, course type, attendance pattern)
- Academic performance (grades at end of 1st and 2nd semesters)
- Financial factors (tuition up to date, scholarship holder)

## Project Structure
```
student-success-prediction/
├── data/                           # Dataset files
│   └── data.csv                    # UCI ML Repository student data (4,424 records)
├── eda/                            # Exploratory Data Analysis
│   ├── eda.ipynb                   # Full analysis notebook
│   ├── eda_summary.md              # Key findings and insights
│   └── eda_summary_draft.md        # Working draft
├── models/                         # Machine Learning Models
│   ├── baseline_models.ipynb       # Full-feature model (76.72% accuracy)
│   ├── early_prediction.ipynb      # Background-only model (64.18% accuracy)
│   ├── WHITEPAPER.md               # 📄 Final technical whitepaper (3 pages)
│   └── REPORT.md                   # Detailed first draft report
├── pics/                           # Visualizations
│   ├── confusion_matrix_final.png
│   ├── feature_importance_final.png
│   ├── feature_importance_background.png
│   └── roc_curves_final.png
├── preprocessing/
│   └── codebook.md                 # Feature definitions and encoding
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/suesuee/student-success-prediction.git
cd student-success-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Load the dataset:
```python
from ucimlrepo import fetch_ucirepo
student_data = fetch_ucirepo(id=697)
```

## Results Summary

### Model Performance

| Model | Accuracy | Dropout Recall | Graduate Recall | Use Case |
|-------|----------|----------------|-----------------|----------|
| **XGBoost (Full)** | 76.72% | 76% (216/284) | 91% (401/442) | After semester grades available |
| **XGBoost (Background)** | 64.18% | 59% (168/284) | 87% (384/442) | At admission (before classes) |
| Dummy Baseline | 49.93% | - | - | Majority class prediction |

### Key Findings

1. **53/47 Split**: Background factors (financial stress, demographics, program selection) contribute 53.2% of predictive power vs. 46.8% from academic performance
2. **Financial Stress Dominates Early Prediction**: Tuition payment status (17.2% importance), scholarship holder (6.9%), and debtor status are top predictors at admission
3. **Second Semester Matters Most**: Second semester grades (14.7% importance) outweigh first semester in predicting outcomes
4. **Engineered Features Work**: Domain-driven features (success_rate, grade_trend, early_warning) rank in top 15 importance

### Two-Stage Intervention Framework

**Stage 1 (Admission)**: Screen incoming students → 59% dropout detection → Financial aid, orientation programs, mentoring

**Stage 2 (After Semester 2)**: Re-assess with grades → 76% dropout detection → Tutoring, academic probation, advising

**Read the full analysis:** [models/WHITEPAPER.md](models/WHITEPAPER.md)

## Usage

### Explore the Analysis

```bash
# Exploratory data analysis
jupyter notebook eda/eda.ipynb

# Full-feature model (76.72% accuracy)
jupyter notebook models/baseline_models.ipynb

# Background-only early prediction (64.18% accuracy)
jupyter notebook models/early_prediction.ipynb
```

### Read the Whitepaper

The complete 3-page technical whitepaper is available at [models/WHITEPAPER.md](models/WHITEPAPER.md).

## Technologies

- **Python 3.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **scikit-learn** - Machine learning models and evaluation
- **XGBoost** - Gradient boosting classifier (best performing model)
- **ucimlrepo** - UCI dataset access
- **Jupyter** - Interactive development environment

## Methodology Highlights

- **Feature Engineering**: Created 4 domain-driven features (success_rate, grade_trend, early_warning, has_zero_grade) that ranked in top 15 importance
- **Class Imbalance Handling**: Used `class_weight='balanced'` to address 50%/32%/18% class distribution
- **Rigorous Evaluation**: 5-fold stratified cross-validation with held-out test set
- **Generalization**: 0.98% gap between CV and test accuracy demonstrates robustness
- **Interpretability**: Feature importance analysis reveals actionable intervention targets

## References

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

## Author

**Sue** - Computer Science (BS), Cal Poly San Luis Obispo

## Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Project completed as part of CSC-466 Knowledge Discovery from Data course at Cal Poly SLO
- Final whitepaper and analysis completed December 2024
---

**For the complete technical analysis, see:** [models/WHITEPAPER.md](models/WHITEPAPER.md)