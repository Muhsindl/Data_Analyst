import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("C:/Users/Muhsin/Desktop/Makine Öğrenmesi Final Ödevi/healthcare-dataset-stroke-data.csv")

print("Bilgiler;\n",df.info())
print("-"*50)
print("Dağılım hakkında;\n",df.describe())
print("-"*50)
print("Verisetinde bulunan sütunlar:\n\n",df.columns)
print("-"*50)
df.columns = df.columns.str.lower().str.replace(' ', '_') #Veriseti düzgün fakat olası bir durumu engellemek için boşluk yerine "_" ifadesi yerleştirdim.
print("String ifadeler için dağılım:\n\n;",df.describe(include="O"))
print("-"*50)
print("Benzersiz değerler:\n",df.nunique())
print("-"*50)

df.drop('id', axis=1, inplace=True) # id sütunu gereksiz olduğu için sildik

print(df.groupby('work_type')['stroke'].mean().sort_values()) #Çalışma tipinin felç durumu ile arasında ki bağlantıyı gösterdik
print("-"*50)
print(df.groupby('residence_type')['stroke'].mean().sort_values()) #İkamet tipinin felç durumu ile arasında ki bağlantıyı gösterdik (Rural-Kırsal / Urban-Kentsel)
print("-"*50)
print(df.groupby('smoking_status')['stroke'].mean().sort_values()) #Sigara içmenin felç ile arasında ki bağlantıyı gösterdik 
print("-"*50)


sns.scatterplot(x = 'avg_glucose_level', y = 'age', hue = 'stroke', data = df) #Ortalama glikoz seviseyine ve yaşa göre felç olma durumu
# plt.savefig("Glikoz Ortalaması vs Yaş etkisi")
plt.waitforbuttonpress()
plt.clf()
plt.close()

sns.scatterplot(x = 'bmi', y = 'age', hue = 'stroke', data = df) #Bmi (vücut kitle indeksi) seviseyine ve yaşa göre felç olma durumu
# plt.savefig("Vücud Kitle İndeksi vs Yaş etkisi")
plt.waitforbuttonpress()
plt.clf()
plt.close()

#Numerik ve Kategorik sütunları ayırdık+
numerical= df.drop(['stroke'], axis=1).select_dtypes('number').columns
categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')


sns.heatmap (df[numerical].corr(), annot=True) #Numerik veriler için korelason gösterdik 
# plt.savefig("Numerik Veriler için korelasyon")
plt.waitforbuttonpress()
plt.clf()
plt.close()


#Ön işleme
df = pd.get_dummies(df, columns=['gender', 'ever_married', #Categorical tipli feature için True/False olarak sınıflandırma yapar
       'work_type', 'residence_type', 'smoking_status',], drop_first=True) 

x_data = df.drop('stroke',axis=1)
y = df['stroke']
 
#Manuel MınMax Scaler (Normalizasyon yaptık)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Verileri train ve test olarak %50-%50 olarak ayırdık
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

x_train.bmi.value_counts(dropna=False)

from sklearn.impute import SimpleImputer #NaN değerlere sütunun ortalamasını  atıyoruz
imputer = SimpleImputer(missing_values=np.nan, strategy="median")

x_train['bmi'] = imputer.fit_transform(x_train['bmi'].values.reshape(-1,1))[:,0]
x_test['bmi'] = imputer.fit_transform(x_test['bmi'].values.reshape(-1,1))[:,0]


print("NaN değer sayıları;\n ",x_train.isnull().sum())
print("-"*50)

#%%
from sklearn.neighbors import KNeighborsClassifier #Sınıflandırma için KNN kullanacağız
# ['euclidean', 'chebyshev', 'minkowski', 'jaccard','manhattan']
knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
y_head=knn.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,classification_report
print("Accuracy Değeri: ",accuracy_score(y_test, y_head))
print("-"*50)
print("R2 Score Değeri: ",r2_score(y_test, y_head))
print("-"*50)
print("Confusion Matris:\n\n ",confusion_matrix(y_test, y_head))
print("-"*50) 
print("Sınıflandırma Sonucu:\n\n ",classification_report(y_test, y_head))

#%%

#Hatasını n sayısını 1-20 aralığında girerek belirleyeceğiz
from sklearn.metrics import recall_score
error_rate = [] 
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_head_i = knn.predict(x_test)
    error_rate.append(1 - recall_score(y_test, y_head_i))
    
plt.plot(range(1,30),error_rate,color='green', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# plt.savefig("Hata oranının k değerine göre değerlendirilmesi")
# plt.waitforbuttonpress()
# plt.clf()
# plt.close()

from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score, cross_validate
model = KNeighborsClassifier(n_neighbors=1)
scores = cross_validate(model, x_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 5) #CrossValidation=5
df_scores = pd.DataFrame(scores, index = range(1, 6))
print("Cross Validation Matris;\n ",df_scores)

#Jaccard ve LogLoss hesaplama
from sklearn.metrics import jaccard_score, log_loss
jaccard = jaccard_score(y_test, y_head, average='weighted')

y_pred_proba = knn.predict_proba(x_test)
logloss = log_loss(y_test, y_pred_proba)

print("Jaccard Skoru:", jaccard)
print("Log Loss Puanı:", logloss)























