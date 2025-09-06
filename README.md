


**Name:** Şerife Gül Korkut  
**Email:** serifegulkorkut@gmail.com

## Project Overview
This project analyzes a Physical Medicine & Rehabilitation dataset (2235 patients, 13 features) to perform thorough EDA and preprocess the data for potential predictive modeling. The target variable is `TedaviSuresi`.

## How to Run the Code
1. Clone the repository:
```bash
git clone https://github.com/gulkorkut/PUSULA_Serife_Gul_KORKUT
````

Ensure the dataset `Talent_Academy_Case_DT_2025.xlsx` is in the project folder.

2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the pipeline:

```bash
python rehab_case_study_pipeline.py
```

The cleaned and processed dataset will be saved as `cleaned_data.csv`.

## Project Structure

```bash
├── case.py       # Main pipeline code
├── case.ipynb       # ipynb version to see plots
├── Talent_Academy_Case_DT_2025.xlsx   # Original dataset
├── cleaned_data.csv                    # Output of the pipeline
├── EDA_Preprocessing_Summary.md        # Summary of EDA & preprocessing
└── README.md                           # This file
```

## Features Processed

* Missing values handled
* Numeric conversion for `TedaviSuresi` and `UygulamaSuresi`
* OneHotEncoding for categorical features
* MultiLabelBinarizer for multi-label columns
* Feature engineering: `Yas_Grubu`, `UygulamaSuresi_per_Seans`
* Scaling (MinMax) for numeric columns

## Notes

* Duplicate rows were removed (928 duplicates)
* EDA included histograms, heatmaps, and target distribution check


