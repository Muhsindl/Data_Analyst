import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri kümesinin okunması
df=pd.read_csv("C:/Users/Muhsin/Desktop/Milli Teknoloji Hamlesi TUBİTAK Ödev/housing.csv")

# Sütun isimleri
print("Sütun isimleri: ")
print(df.columns)
print("-"*50)

# Veri kümesi dağılımı
print("Veri kümesi dağılımı: ")
print(df.describe())
print("-"*50)

print("Bilgi: ")
print(df.info())
print("-"*50)
# object tipinde tek sütun var bu sütun ise "ocean_proximity"

print("Sütunda ki çeşitli değerler: ")
print(df.ocean_proximity.unique())
print("-"*50)
# 'NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND' olmak üzere 4 çeşit str değer içeriyor







# Null Değer kontrolü
print("NaN değer var mı?")
print(df.isnull().sum())
print("-"*50)
# total_bedrooms sütununda 207 adet NaN değer var






