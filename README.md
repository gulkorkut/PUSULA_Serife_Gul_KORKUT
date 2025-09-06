# EDA & Preprocessing Summary
**Name:** Şerife Gül Korkut  
**Email:** gulkorkut@example.com  

## 1. Overview
This project analyzes a Physical Medicine & Rehabilitation dataset consisting of 2235 patients and 13 features. The goal was to perform in-depth EDA and preprocess the data to make it ready for potential predictive modeling, with **TedaviSuresi** as the target variable.

## 2. Data Loading
- The dataset was loaded from `Talent_Academy_Case_DT_2025.xlsx`.
- Shape: 2235 rows × 13 columns.

## 3. Exploratory Data Analysis (EDA)
### 3.1 Data Types & Missing Values
- Numerical: `HastaNo`, `Yas`, `TedaviSuresi`, `UygulamaSuresi`
- Categorical / Object: `Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum`, `KronikHastalik`, `Alerji`, `Tanilar`, `TedaviAdi`, `UygulamaYerleri`
- Missing values:
  - `Cinsiyet`: 169
  - `KanGrubu`: 675
  - `KronikHastalik`: 611
  - `Alerji`: 944
  - `Tanilar`: 75
  - `UygulamaYerleri`: 221
  - `Bolum`: 11
- Duplicates: 928 rows → removed, final dataset: 1307 rows.

### 3.2 Target Distribution
- Target variable: `TedaviSuresi`
- Distribution: roughly skewed; visualized using histogram & KDE.
- Outliers identified via boxplot.

### 3.3 Categorical & Multi-label Variables
- Top 10 chronic diseases: Aritmi, Hiportiroidizm, Limb-Girdle Musküler Distrofi, Astım, Hipertiroidizm, Myastenia gravis, Diyabet, Duchenne Musküler Distrofisi, Fascioscapulohumeral Distrofi, Kalp yetmezliği
- Top 10 allergies: Polen, POLEN, Toz, TOZ, NOVALGIN, ARVELES, CORASPIN, Sucuk, Yer Fıstığı, SUCUK
- Top 10 diagnoses: DORSALJİ, DİĞER, tanımlanmamış, Omuzun darbe sendromu, İntervertebral disk bozuklukları, LUMBOSAKRAL BÖLGE, SERVİKOTORASİK BÖLGE, SERVİKAL BÖLGE, Eklem ağrısı, Dorsalji
- Top 10 application sites: Bel, Boyun, Diz, Sol Omuz Bölgesi, Sağ Omuz Bölgesi, Sırt, Sol El Bilek Bölgesi, Sağ Ayak Bileği Bölgesi, Sol Ayak Bölgesi, Tüm Vücut Bölgesi

### 3.4 Numeric Features
- `Yas` mean: 47.3, std: 15.2, min: 2, max: 92
- `UygulamaSuresi` converted to integer for numeric operations

## 4. Preprocessing Steps
1. **Removed duplicates** to ensure unique patient records.
2. **Filled missing values**:
   - Numerical (`Yas`): median
   - Categorical: most frequent
   - Multi-label: empty string for missing
3. **Converted numeric columns**: `TedaviSuresi`, `UygulamaSuresi`
4. **OneHotEncoding**: `Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum`, `TedaviAdi`
5. **MultiLabelBinarizer**: `KronikHastalik`, `Alerji`, `Tanilar`, `UygulamaYerleri`
6. **Feature Engineering**:
   - `Yas_Grubu` (Age groups)
   - `UygulamaSuresi_per_Seans` = `UygulamaSuresi` / `TedaviSuresi`
7. **Scaling / Normalization**:
   - `Yas`, `UygulamaSuresi`, `UygulamaSuresi_per_Seans` scaled using MinMaxScaler.

## 5. Outcome
- Cleaned, encoded, and normalized dataset ready for predictive modeling.
- Saved as `cleaned_data.csv`.


I also added .ipynb version to show plots
