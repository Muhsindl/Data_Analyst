import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Verisetinin okunması
df=pd.read_csv("C:/Users/Muhsin/Desktop/panic_attack_dataset.csv")
# profile = ProfileReport(df, title="Profiling Report")
# profile.to_file("output.html")

#---------------------------- Veri Analizi ----------------------------

print("Sütunların isimler: \n",df.columns)

print("Veriseti hakkında bilgi: \n",df.info())

print("Toplam NaN değer sayıları sütun bazında: \n",df.isnull().sum())

print("Eksik (Medical_History) sütunda ki veri çeşitleri: ", df.Medical_History.unique())

"""
Anxiety -----> Endişe, kaygı gibi anskiyete
PTSD --------> Tramva sonrası stres bozukluğu 
Depression --> Depresyon belirtisi
nan ---------> Eksik veriler

"""
print("-"*50)

# Object türündeki sütunları bulmak için yapıyoruz
obj = []
for col in df.columns:
    if df[col].dtype == "object":
        obj.append(col)

# Nesne tipinde ki sütunların dağılımı için yapıyoruz
obj_Series = pd.Series(obj)

for i in obj_Series:
    print(f"{i} : {df[i].unique()}")

# Trigger sütununda "Unknown" değerlerini NaN yapıyoruz
df["Trigger"] = df["Trigger"].replace("Unknown", np.nan)

# NaN değerleri barındıran satırları siliyoruz
df.dropna(subset=["Trigger"], inplace=True)

# Trigger sütununu sayısal değerlere dönüştürüyoruz
df["Trigger"] = [
    0 if i == "Caffeine" else 
    1 if i == "Stress" else 
    2 if i == "PTSD" else 
    3 for i in df["Trigger"]
]


df.Gender=[ 0 if i=="Female" else 1 if i=="Male" else 2 for i in df.Gender]

df.Sweating=[ 1 if i=="Yes" else 0 for i in df.Sweating]
df.Shortness_of_Breath=[ 1 if i=="Yes" else 0 for i in df.Shortness_of_Breath]
df.Dizziness=[ 1 if i=="Yes" else 0 for i in df.Dizziness]
df.Chest_Pain=[ 1 if i=="Yes" else 0 for i in df.Chest_Pain]
df.Trembling=[ 1 if i=="Yes" else 0 for i in df.Trembling]
df.Medication=[ 1 if i=="Yes" else 0 for i in df.Medication]
df.Smoking=[ 1 if i=="Yes" else 0 for i in df.Smoking]
df.Therapy=[ 1 if i=="Yes" else 0 for i in df.Therapy]

df.dropna(subset=["Medical_History"], inplace=True)
df.Medical_History=[ 0 if i=="Anxiety" else 1 if i=="PTSD" else 3 for i in df.Medical_History]

#---------------------------- Veri Görselleştirme ----------------------------


print(f"Minimum Yaş: {df.Age.min()}")
print(f"Maximum Yaş: {df.Age.max()}")
print(f"Ortalama Yaş: {df.Age.mean()}")
sns.boxplot(data=df,x="Age",hue=True)
plt.show()
plt.waitforbuttonpress()
plt.clf()
plt.close()

sns.barplot(x=df["Gender"].value_counts().index, y=df["Gender"].value_counts().values)
plt.title("Cinsiyet Dağılımı")
plt.xlabel("Gender (0: Female, 1: Male, 2: Other)")
plt.ylabel("Count")
plt.show()
plt.waitforbuttonpress()
plt.clf()
plt.close()

#-------------------------- Model Eğitim hazırlığı --------------------------

#Verilerin x ve y olarak alınması
x_data=df.drop(["Panic_Score","ID"],axis=1).values
y=df["Panic_Score"].values.reshape(-1,1)


# Normalizasyon uygulanması
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=71)

#---------------------------- Model Eğitimi (KNN) ----------------------------

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# K değerlerini denemek için bir liste oluşturuyoruz
k_values = range(1, 51)

# Performans sonuçlarını kaydetmek için bir liste
mae_scores = []
r2_scores = []

# Döngü ile farklı k değerlerini deniyoruz
for k in k_values:
    kNN_reg = KNeighborsRegressor(n_neighbors=k)
    kNN_reg.fit(x_train, y_train)
    y_head = kNN_reg.predict(x_test)
    
    # Performans metriklerini hesaplıyoruz
    mae = mean_absolute_error(y_test, y_head)
    r2 = r2_score(y_test, y_head)
    
    # Sonuçları kaydediyoruz
    mae_scores.append(mae)
    r2_scores.append(r2)

# Performans sonuçlarını görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(k_values, mae_scores, label="Mean Absolute Error (MAE)", marker='o')
plt.plot(k_values, r2_scores, label="R-squared (R2)", marker='o')
plt.xlabel("K Değerleri")
plt.ylabel("Performans")
plt.title("KNN Modeli için K Değerlerinin Performansa Etkisi")
plt.legend()
plt.grid(True)
plt.show()

# En iyi k değerini bulma
best_k_mae = k_values[np.argmin(mae_scores)]
best_k_r2 = k_values[np.argmax(r2_scores)]
print(f"En düşük MAE için en iyi k: {best_k_mae}")
print(f"En yüksek R2 için en iyi k: {best_k_r2}")

