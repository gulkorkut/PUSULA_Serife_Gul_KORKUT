# rehab_case_study_pipeline.py

# -------------------------
# 1. Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, MinMaxScaler, StandardScaler

# -------------------------
# Rehab Pipeline Class
# -------------------------
class RehabPipeline:
    def __init__(self, path="Talent_Academy_Case_DT_2025.xlsx"):
        self.path = path
        self.df = None
        self.scaler = None

    # -------------------------
    # 2. Data Loading
    # -------------------------
    def load_data(self):
        self.df = pd.read_excel(self.path)
        print(f"Veri yüklendi: {self.df.shape[0]} satır, {self.df.shape[1]} kolon")
        return self.df

    # -------------------------
    # 3a. EDA
    # -------------------------
    def basic_eda(self):
        df = self.df
        print("\nVeri tipleri:")
        print(df.dtypes)
        
        print("\nEksik değerler:")
        print(df.isnull().sum())
        
        print(f"\nDuplicate satır sayısı: {df.duplicated().sum()}")
        
        print("\nTemel istatistikler:")
        print(df.describe(include='all'))
        
        # Histogramlar
        plt.figure(figsize=(8,5))
        sns.histplot(df['Yas'], bins=20)
        plt.title("Yaş Dağılımı")
        plt.show()
        
        plt.figure(figsize=(8,5))
        sns.histplot(df['TedaviSuresi'].apply(lambda x: int(str(x).split()[0])), bins=20)
        plt.title("Tedavi Süresi Dağılımı (Seans)")
        plt.show()
        
        # heatmap
        numeric_cols = ['Yas','UygulamaSuresi']
        df_numeric = df.copy()
        df_numeric['UygulamaSuresi'] = pd.to_numeric(df_numeric['UygulamaSuresi'].str.extract('(\d+)')[0], errors='coerce')
        sns.heatmap(df_numeric[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Numeric Feature Correlation")
        plt.show()

        # Target distribution
        plt.figure(figsize=(8,5))
        sns.histplot(df['TedaviSuresi'], bins=20, kde=True)
        plt.title("Target (TedaviSuresi) Distribution")
        plt.show()

    # -------------------------
    # 3b. Multi-label Frequency
    # -------------------------
    def multi_label_frequency(self):
        df = self.df
        mlb_cols = ['KronikHastalik','Alerji','Tanilar','UygulamaYerleri']
        for col in mlb_cols:
            all_items = df[col].dropna().apply(lambda x: [i.strip() for i in str(x).split(',')]).explode()
            freq = all_items.value_counts().head(10)
            print(f"\nTop 10 {col} values:")
            print(freq)

    # -------------------------
    # 4. Remove Duplicate
    # -------------------------
    def remove_duplicates(self):
        df = self.df
        print(f"Duplicate sayısı: {df.duplicated().sum()}")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicate sonrası satır sayısı: {df.shape[0]}")
        self.df = df
        return df

    # -------------------------
    # 5. Fill Missing Values
    # -------------------------
    def fill_missing_values(self):
        df = self.df
        # Numerical Columns
        num_cols = ['Yas']
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

        # Categoric Columns
        cat_cols = ['Cinsiyet','KanGrubu','Uyruk','Bolum']
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        # Empty String for Multi-label Columns
        mlb_cols = ['KronikHastalik','Alerji','Tanilar','UygulamaYerleri']
        for col in mlb_cols:
            df[col] = df[col].fillna('')
        
        self.df = df
        return df

    # -------------------------
    # 6. Encoding
    # -------------------------
    def encode_features(self):
        df = self.df
        # OneHotEncoding
        ohe_cols = ['Cinsiyet','KanGrubu','Uyruk','Bolum','TedaviAdi']
        ohe = OneHotEncoder(sparse_output=False)
        encoded = ohe.fit_transform(df[ohe_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
        df = pd.concat([df.drop(columns=ohe_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Encoding Multi-label Columns
        mlb_cols = ['KronikHastalik','Alerji','Tanilar','UygulamaYerleri']
        for col in mlb_cols:
            mlb = MultiLabelBinarizer()
            split_col = df[col].apply(lambda x: [i.strip() for i in x.split(',')] if x != '' else [])
            encoded_multi = mlb.fit_transform(split_col)
            encoded_multi_df = pd.DataFrame(encoded_multi, columns=[f"{col}_{c}" for c in mlb.classes_])
            df = pd.concat([df.drop(columns=[col]), encoded_multi_df], axis=1)
        
        self.df = df
        return df

    # -------------------------
    # 7. Convert Numeric
    # -------------------------
    def convert_numeric(self):
        df = self.df
        df['TedaviSuresi'] = df['TedaviSuresi'].str.extract('(\d+)').astype(int)
        df['UygulamaSuresi'] = df['UygulamaSuresi'].str.extract('(\d+)').astype(int)
        self.df = df
        return df

    # -------------------------
    # 8. Feature Engineering
    # -------------------------
    def feature_engineering(self):
        df = self.df
        bins = [0,30,50,70,100]
        labels = ['Genç','Orta Yaş','Yaşlı','Çok Yaşlı']
        df['Yas_Grubu'] = pd.cut(df['Yas'], bins=bins, labels=labels)
        self.df = df
        return df

    # -------------------------
    # 8a. Additional Features
    # -------------------------
    def additional_features(self):
        df = self.df
        df['UygulamaSuresi_per_Seans'] = df['UygulamaSuresi'] / df['TedaviSuresi']
        self.df = df
        return df

    # -------------------------
    # 8b. Scaling / Normalization
    # -------------------------
    def scale_numeric(self, method="minmax"):
        df = self.df
        num_cols = ['Yas','UygulamaSuresi','UygulamaSuresi_per_Seans']
        if method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError("method should be 'minmax' or 'standard'")
        
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        self.df = df
        return df

    # -------------------------
    # 9. Plots
    # -------------------------
    def plot_histograms(self):
        df = self.df
        plt.figure(figsize=(8,5))
        sns.histplot(df['Yas'], bins=20)
        plt.title("Yaş Dağılımı")
        plt.show()

        plt.figure(figsize=(8,5))
        sns.histplot(df['TedaviSuresi'], bins=20)
        plt.title("Tedavi Süresi Dağılımı")
        plt.show()

        plt.figure(figsize=(8,5))
        sns.countplot(x='Yas_Grubu', data=df)
        plt.title("Yaş Gruplarına Göre Hasta Sayısı")
        plt.show()

    # -------------------------
    # 10. Save Cleaned Data
    # -------------------------
    def save_cleaned_data(self, path="cleaned_data.csv"):
        self.df.to_csv(path, index=False)
        print(f"Temizlenmiş veri kaydedildi: {path}")

    # -------------------------
    # 11. Run Full Pipeline
    # -------------------------
    def run(self):
        self.load_data()
        self.basic_eda()
        self.multi_label_frequency()
        self.remove_duplicates()
        self.fill_missing_values()
        self.encode_features()
        self.convert_numeric()
        self.feature_engineering()
        self.additional_features()
        self.scale_numeric(method="minmax")  # min-max scaling ekledik
        self.plot_histograms()
        self.save_cleaned_data()


# -------------------------
# Script Execution
# -------------------------
if __name__ == "__main__":
    pipeline = RehabPipeline()
    pipeline.run()
