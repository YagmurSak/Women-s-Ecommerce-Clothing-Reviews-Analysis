import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
import random


# Uyarıları kapatma
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df_ = pd.read_csv(r"C:\Users\ASUS\PycharmProjects\pythonProject\Aygaz_Bootcamp\Womens Clothing E-Commerce Reviews.csv")

df = df_

df.head()

df.isnull().sum()
##################################################################################
def add_random_missing_values(dataframe: pd.DataFrame,
                              missing_rate: float = 0.05) -> pd.DataFrame:
    """Turns random values to NaN in a DataFrame.

    To use this function, you need to import pandas, numpy and random libraries.

    Args:
        dataframe (pd.DataFrame): DataFrame to be processed.
        missing_rate (float): Percentage of missing value rate in float format. Defaults 0.05

    Returns:
        df_missing (pd.DataFrame): Processed DataFrame object.

    """
    # Get copy of dataframe
    df_missing = dataframe.copy()

    # Obtain size of dataframe and number total number of missing values
    df_size = dataframe.size
    num_missing = int(df_size * missing_rate)

    # Get random row and column indexes to turn them NaN
    for _ in range(num_missing):
        row_idx = random.randint(0, dataframe.shape[0] - 1)
        col_idx = random.randint(0, dataframe.shape[1] - 1)

        df_missing.iat[row_idx, col_idx] = np.nan

    return df_missing

df_missing = add_random_missing_values(df)

df.isnull().sum().sum()
df_missing.isnull().sum().sum()

df = df_missing
##################################################################################
df.head()
df.info()
##################################################################################
def unique_values(dataframe):

    for col in df.columns:
        unique_values = df[col].unique()  # Sütundaki benzersiz değerleri alır
        print(f"{col} sütununun benzersiz değerleri:{df[col].nunique()}")
        print(unique_values)
        print("\n" + "#" * 50 + "\n")  # Her sütunun sonunda ayırıcı ekler

unique_values(df)


##################################################################################

print(df.info())  # Veri tipi ve eksik değerleri görme
print(df.describe())  # Sayısal sütunlar için özet istatistikler
print(df.isnull().sum() / len(df) * 100)  # Eksik değer yüzdelerini inceleme

##################################################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    float_columns = dataframe.select_dtypes(include=['float64'])
    print("##################### Quantiles #####################")
    print(float_columns.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)} {cat_cols}')
    print(f'num_cols: {len(num_cols)} {num_cols}')
    print(f'cat_but_car: {len(cat_but_car)} {cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)} {num_but_cat}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################################################################

# KATEGORİK DEĞİŞKENLERİN ANALİZİ
def cat_summary(dataframe, col_names, plot=False):

    if not isinstance(col_names, list):
        col_names = [col_names]


    for col_name in col_names:
        print(f"Analiz edilen sütun: {col_name}")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.title(f"{col_name} için Frekans Grafiği")
            plt.show(block = True)

cat_summary(df, cat_cols, plot=False)

##################################################################################

# NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col, plot=False):
    # Sayısal değişkenlerin özet istatistiklerini yazdır
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        # Histogram çiz
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)  # Birinci grafiği çiz
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Histogram")

        # Box plot çiz
        plt.subplot(1, 2, 2)  # İkinci grafiği çiz
        dataframe.boxplot(column=numerical_col)
        plt.title(f"{numerical_col} Box Plot")

        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)



##################################################################################


##################################################################################

#  Eksik Değer Analizi
print("\nEksik Değer Analizi:")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / df.shape[0]) * 100
missing_df = pd.DataFrame({'Eksik Değer Sayısı': missing_values, 'Eksik Değer Oranı (%)': missing_percentage})
print(missing_df[missing_df['Eksik Değer Sayısı'] > 0])  # Eksik değer oranları


# Eksik değerleri görselleştir
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Eksik Değerlerin Görselleştirilmesi")
plt.show(block=True)

##################################################################################

# Sadece sayısal sütunları seçme
numeric_df = df[num_cols]

# Korelasyon matrisi oluşturma ve görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="magma")
plt.title("Sayısal Sütunlar Arasındaki Korelasyon Matrisi")
plt.show(block=True)

corr = df[num_cols].corr()

##################################################################################
#  Sayısal Değişkenler Arası İlişkiler
# Çift değişkenli dağılım için scatter plot matrisi
sns.pairplot(df.select_dtypes(include=[np.number]), diag_kind='kde', plot_kws={'alpha':0.7})
plt.suptitle("Sayısal Değişkenler Arası İlişkiler", y=1.02)
plt.show(block=True)

##################################################################################


# Kategorik ve Sayısal Değişkenler Arasındaki Kırılımlar
# pivot table ile kategorik değişkenlerin sayısal değerlere göre özet istatistikleri
for col in df.select_dtypes(include=['object']).columns:
    pivot = pd.pivot_table(df, index=col, values=df.select_dtypes(include=[np.number]).columns, aggfunc=['mean', 'median', 'std'])
    print(f"\n{col} Değişkenine Göre Sayısal Değişkenlerin Kırılımı:")
    print(pivot.head(10))


##################################################################################


# Özelliklerin Normalite Kontrolü
# Sayısal sütunların normal dağılıma uyup uymadığını incele
from scipy.stats import shapiro

for col in df.select_dtypes(include=[np.number]).columns:
    stat, p = shapiro(df[col].dropna())
    print(f"{col} için Shapiro Testi: Test İstatistiği = {stat}, p-değeri = {p}")
    if p > 0.05:
        print(f"{col} normal dağılıma uygun.")
    else:
        print(f"{col} normal dağılıma uygun değil.")



##################################################################################
#Aykırı Değerler

def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit
##################################################################################
# Aykırı değer Kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "price":
      print(col, check_outlier(df, col))

##################################################################################
# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, grab_outliers(df, col))

##################################################################################
# Aykırı Değerlerin Baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "price":
        replace_with_thresholds(df,col)


for col in num_cols:
    if col != "price":
      print(col, check_outlier(df, col))

##################################################################################


# NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col, plot=False):
    # Sayısal değişkenlerin özet istatistiklerini yazdır
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        # Histogram çiz
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)  # Birinci grafiği çiz
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Histogram")

        # Box plot çiz
        plt.subplot(1, 2, 2)  # İkinci grafiği çiz
        dataframe.boxplot(column=numerical_col)
        plt.title(f"{numerical_col} Box Plot")

        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

##################################################################################
#  Eksik Değer Analizi
print("\nEksik Değer Analizi:")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / df.shape[0]) * 100
missing_df = pd.DataFrame({'Eksik Değer Sayısı': missing_values, 'Eksik Değer Oranı (%)': missing_percentage})
print(missing_df[missing_df['Eksik Değer Sayısı'] > 0])  # Eksik değer oranları


# Eksik değerleri görselleştir
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Eksik Değerlerin Görselleştirilmesi")
plt.show(block=True)

##################################################################################
df.head()
df["Clothing ID"].unique()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.isnull().sum()

# Clothing ID sütunundaki eksik değerleri 0 ile doldurma
df['Clothing ID'] = df['Clothing ID'].fillna(0)

#"Unknown" gibi bir placeholder kullandık
df['Title'].fillna("Unknown", inplace=True)

#"No Review" gibi bir placeholder kullandık
df['Review Text'].fillna("No Review", inplace=True)

#Eksik değerler sıfır (0) ile dolduruldu.
df['Unnamed: 0'].fillna(0, inplace=True)

#Gruplandırılmış(Department Name Kırılımında) mod kullanıldı
df['Class Name'] = df['Class Name'].fillna(
    df.groupby('Department Name')['Class Name'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
)
# Eğer hala eksik değer varsa Division Name bazında doldurma
if df['Class Name'].isnull().sum() != 0:
    df['Class Name'] = df['Class Name'].fillna(
        df.groupby('Division Name')['Class Name'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    )

df['Department Name'] = df['Department Name'].fillna(
    df.groupby('Class Name')['Department Name'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
)
# Eğer hala eksik değer varsa Division Name bazında doldurma
if df['Department Name'].isnull().sum() != 0:
    df['Department Name'] = df['Department Name'].fillna(
        df.groupby('Division Name')['Department Name'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    )

# Division Name eksik değerlerini Department Name ve Class Name'e göre gruplandırılmış mod ile doldurma
df['Division Name'] = df['Division Name'].fillna(
    df.groupby(['Department Name', 'Class Name'])['Division Name'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    )
)

# Eğer hala eksik değer varsa sadece Department Name'e göre doldurma
if df['Division Name'].isnull().sum() != 0:
    df['Division Name'] = df['Division Name'].fillna(
        df.groupby('Department Name')['Division Name'].transform(
            lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
        )
    )
# Positive Feedback Count eksik değerlerini Department Name ve Class Name'e göre gruplandırılmış ortalama değerlerle doldurma
df['Positive Feedback Count'] = df['Positive Feedback Count'].fillna(
    df.groupby(['Department Name', 'Class Name'])['Positive Feedback Count'].transform('mean')
)

# Eğer hala eksik değer varsa Class Name bazında doldur
if df['Positive Feedback Count'].isnull().sum() != 0:
    df['Positive Feedback Count'] = df['Positive Feedback Count'].fillna(
        df.groupby(['Division Name'])['Positive Feedback Count'].transform('mean')
    )

# Age sütunundaki eksik değerleri Class Name'e göre gruplandırılmış medyan ile doldurma
df['Age'] = df['Age'].fillna(
    df.groupby('Class Name')['Age'].transform('median')
)


from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# Rating sütunu için KNN ile eksik değer doldurma

# Rating sütununun ölçeklendirilmesi
scaler = RobustScaler()
rating_scaled = pd.DataFrame(scaler.fit_transform(df[['Rating']]), columns=['Rating'])

# KNN ile eksik değer doldurma
imputer = KNNImputer(n_neighbors=5)
rating_imputed = pd.DataFrame(imputer.fit_transform(rating_scaled), columns=['Rating'])

# Orijinal ölçeğe geri döndürme
rating_final = pd.DataFrame(scaler.inverse_transform(rating_imputed), columns=['Rating'])

# Orijinal veri setinde Rating sütununu güncelleme
df['Rating'] = rating_final['Rating']

# Eksik değerlerin kontrolü
rating_missing_after_knn = df['Rating'].isnull().sum()

rating_missing_after_knn


# Class Name ve Rating kırılımında mod ile doldurma
df['Recommended IND'] = df['Recommended IND'].fillna(
    df.groupby(['Class Name', 'Rating'])['Recommended IND'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else 0
    )
)

# Hala eksik varsa genel mod ile doldur
if df['Recommended IND'].isnull().sum() > 0:
    df['Recommended IND'].fillna(df['Recommended IND'].mode()[0], inplace=True)


# Eğer hala eksik değer varsa bu satırları sil
if df.isnull().sum().sum() > 0:
    df = df.dropna()

df.isnull().sum()