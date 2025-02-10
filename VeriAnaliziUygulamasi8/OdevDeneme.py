"""
California Housing Veri Seti Üzerinde PySpark ile Lineer Regresyon Uygulaması

Bu kod, California Housing veri setini kullanarak PySpark ile bir Lineer Regresyon modeli oluşturur.
Veri ön işleme, özellik mühendisliği, model eğitimi ve değerlendirme adımlarını içerir.
"""

# Gerekli kütüphanelerin yüklenmesi (Eğer PySpark yüklü değilse çalıştırın)
!pip install pyspark

# Gerekli kütüphanelerin içe aktarılması
from pyspark.sql import SparkSession  # Spark oturumu oluşturmak için
from pyspark.ml.feature import VectorAssembler  # Özellikleri vektör haline getirmek için
from pyspark.ml.regression import LinearRegression  # Lineer regresyon modeli için
from pyspark.ml.evaluation import RegressionEvaluator  # Model değerlendirme metrikleri için
from pyspark.sql.functions import mean  # Ortalama hesaplamak için
from prettytable import PrettyTable  # Çıktıları tablo formatında göstermek için

# Spark oturumunu başlatma
spark = SparkSession.builder.appName("CaliforniaHousing").getOrCreate()

# Veri kümesini yükleme (Dosya yolunu kendi sisteminize göre değiştirin)
file_path = "/housing.csv"  # Veri kümesinin bulunduğu yol
df = spark.read.csv(file_path, header=True, inferSchema=True)  # CSV dosyasını oku ve şemayı otomatik algıla

# **Veri Keşfi ve Ön Analiz**
print("Dataset schema:")
df.printSchema()  # Veri kümesindeki sütunların türlerini ve yapılarını göster

print("\nNumber of rows:", df.count())  # Veri kümesindeki toplam satır sayısını göster

print("\nFirst few rows:")
df.show(5)  # İlk 5 satırı göstererek veri hakkında genel bir fikir edin

# **Eksik Değerlerin İşlenmesi**
# 'total_bedrooms' sütununun ortalama değerini hesapla
mean_total_bedrooms = df.select(mean('total_bedrooms')).collect()[0][0]

# Eksik değerleri hesaplanan ortalama ile doldur
df = df.na.fill({'total_bedrooms': mean_total_bedrooms})

# **Özellik Mühendisliği ve Seçimi**
# Kullanılacak özellik sütunlarını belirle
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']

# Belirlenen sütunları tek bir 'features' sütununa dönüştür
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)  # Yeni bir "features" sütunu oluştur

# **Veri Kümesinin Eğitim ve Test Olarak Bölünmesi**
# Veriyi %80 eğitim, %20 test olacak şekilde ikiye ayır
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# **Lineer Regresyon Modelinin Eğitilmesi**
lr = LinearRegression(featuresCol="features", labelCol="median_house_value")
model = lr.fit(train_data)  # Modeli eğitim verisi üzerinde eğit

# **Modelin Test Verisi Üzerinde Değerlendirilmesi**
# Test verisi üzerinde tahmin yap
predictions = model.transform(test_data)

# Modelin performansını değerlendirmek için Kök Ortalama Kare Hatası (RMSE) hesapla
evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

# Sonuçları tablo formatında göstermek için PrettyTable kullan
results_table = PrettyTable()
results_table.field_names = ["Metric", "Value"]
results_table.add_row(["Root Mean Squared Error (RMSE)", round(rmse, 2)])
results_table.add_row(["Intercept", round(model.intercept, 2)])

# Katsayıları da tabloya ekle
coefficients = model.coefficients
table_coeff = PrettyTable()
table_coeff.field_names = ["Feature", "Coefficient"]
for feature, coeff in zip(features, coefficients):
    table_coeff.add_row([feature, round(coeff, 2)])

# Sonuçları ve katsayıları yazdır
print(results_table)
print("\nModel Coefficients:")
print(table_coeff)

# **Spark Oturumunu Sonlandırma**
spark.stop()
