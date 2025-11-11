# Student Success Prediction

Machine learning classifier to predict student academic outcomes (dropout, enrolled, or graduate) for early intervention and student support programs.

## Project Overview

This project uses student enrollment and academic performance data to build a classification model that predicts whether students will dropout, remain enrolled, or successfully graduate. Early identification of at-risk students enables universities to:
- Implement targeted intervention strategies
- Provide academic support resources
- Improve retention rates
- Enhance student success outcomes

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
├── data/                      # Dataset files
│   └── data.csv
├── eda/                       # Exploratory Data Analysis
│   ├── eda.ipynb
│   └── eda_summary.md
├── models/                    # Model training (coming soon)
├── requirements.txt           # Python dependencies
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

## Usage

Open and run the Jupyter notebook for exploratory data analysis:
```bash
jupyter notebook eda/eda.ipynb
```

## Technologies

- **Python 3.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **scikit-learn** - Machine learning models and evaluation
- **ucimlrepo** - UCI dataset access
- **Jupyter** - Interactive development environment

## Key Findings

*Coming soon after EDA*

## Model Performance

*Coming soon - will include comparison of multiple classification algorithms*

## Future Work

- Implement multiple classification algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- Handle class imbalance with SMOTE or class weights
- Feature engineering and selection
- Hyperparameter tuning
- Deploy model as a web application for real-time predictions

## References

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

## Author

Sue - Cal Poly San Luis Obispo, Computer Science (BS)

## Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Project completed as part of CSC-466 Knowledge Discovery course at Cal Poly
```