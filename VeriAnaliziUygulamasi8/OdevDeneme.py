# Gerekli kütüphanelerin içe aktarılması
from pyspark.sql import SparkSession  # Spark oturumu oluşturmak için kullanılır
from pyspark.ml.feature import VectorAssembler  # Özellikleri vektör haline getirmek için kullanılır
from pyspark.ml.regression import LinearRegression  # Lineer regresyon modeli için
from pyspark.ml.evaluation import RegressionEvaluator  # Model değerlendirme metrikleri için
from pyspark.sql.functions import mean
from prettytable import PrettyTable  # Çıktıları daha okunaklı tablo formatında göstermek için

# Spark oturumunu başlatıyoruz
spark = SparkSession.builder.appName("CaliforniaHousing").getOrCreate()

# Veri kümesini yükleme (Dosya yolunu kendi sisteminize göre değiştirin)
file_path = "/housing.csv"  # Veri kümesinin bulunduğu yol
# CSV dosyasını okuma, başlık bilgisini kullan ve veri türlerini otomatik algıla
df = spark.read.csv(file_path, header=True, inferSchema=True)

# **Veri Keşfi ve Ön Analiz**
print("Dataset schema:")
df.printSchema()  # Veri kümesindeki sütunların türlerini ve yapılarını gösterir

print("\nNumber of rows:", df.count())  # Veri kümesindeki toplam satır sayısını gösterir

print("\nFirst few rows:")
df.show(5)  # İlk 5 satırı göstererek veri hakkında genel bir fikir edinmemizi sağlar


# 'total_bedrooms' sütununun ortalama değerini hesapla
mean_total_bedrooms = df.select(mean('total_bedrooms')).collect()[0][0]

# Eksik değerleri hesaplanan ortalama ile doldur
# Eğer 'total_bedrooms' sütununda eksik veri varsa, bu eksik değerleri ortalama ile dolduruyoruz
df = df.na.fill({'total_bedrooms': mean_total_bedrooms})


# Kullanılacak özellik sütunlarını belirliyoruz
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
            'total_bedrooms', 'population', 'households', 'median_income']
# Belirlenen sütunları tek bir 'features' sütununa dönüştür
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)  # Yeni bir "features" sütunu oluşturulur

# **Veri Kümesinin Eğitim ve Test Olarak Bölünmesi**
# Veriyi %80 eğitim, %20 test olacak şekilde ikiye ayırıyoruz
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# **Lineer Regresyon Modelinin Eğitilmesi**
lr = LinearRegression(featuresCol="features", labelCol="median_house_value")
model = lr.fit(train_data)  # Modeli eğitim verisi üzerinde eğitiyoruz

# **Modelin Test Verisi Üzerinde Değerlendirilmesi**
# Test verisi üzerinde tahmin yapıyoruz
predictions = model.transform(test_data)

# Modelin performansını değerlendirmek için Kök Ortalama Kare Hatası (RMSE) hesaplıyoruz
evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
