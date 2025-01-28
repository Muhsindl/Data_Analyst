import pandas as pd

df=pd.read_csv("C:/Users/Muhsin/Desktop/Milli Teknoloji Hamlesi TUBİTAK Ödev/housing.csv")

print("Bilgi: ")
print(df.info())
print("-"*50)
print(df.describe())
print("NaN değer var mı?")
print(df.isnull().sum())
