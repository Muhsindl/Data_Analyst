import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Verisetinin okunması
df=pd.read_csv("C:/Users/Muhsin/Desktop/panic_attack_dataset.csv")

# Verisetinin profilinin çıkarılması
"""
# output.html dosyası oluşturuldu ve veri seti hakkında detaylı bilgileri içeriyor

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("output.html")
"""

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

# Trigger sütununda Unknown değerleri görüyoruz bu değerler NaN kontrolünde ulaşılamamıştı
print("Toplam (Trigger) sütununda ki Unknown değer sayısı: ",len(df.Trigger[df.Trigger=="Unknown"]))

# Unknown etkisini en aza indirmek için 0 değeri verdim ve diğerlerini de sayısal formata çevirdim
df.Trigger = [
    0 if i == "Unknown" else 
    1 if i == "Caffeine" else 
    2 if i == "Stress" else 
    3 if i == "PTSD" else 
    4 for i in df.Trigger
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
