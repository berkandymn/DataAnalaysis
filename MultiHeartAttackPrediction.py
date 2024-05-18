#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import sklearn.metrics as mt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

#modeller
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier



#pandas ile .csv formatındaki veriyi okuyalım
veriler=pd.read_csv('medicaldataset.csv')
veriler.head()


#Veri setinin özelliklerine göz atalım
print(f'Shape of data: {veriler.shape}\nNumber of Columns: {len(veriler.columns)}\nSize of Dataset: {veriler.size}')

veriler.info()

#veri setinin her sütununa ait özet bilgileri
veriler. describe()

#Her sutunda bulunan null veya eksik verileri gözlemleyelim
veriler.isna().sum()

#Her sütuna ait eşsiz tekrar etmeyen verilerin sayılarına bakalım
for col in veriler.columns:
    print('Unique data in', col, ':', veriler[col].value_counts().nunique())

#veri setini kopyalayalım ve tahmin edilecek bölümü ayıralım
cpVeriler=veriler.copy()
cpVeriler.pop('Result')
cpVeriler.head()


#daha önce sayısal veriye dönüştürülmüş cinsiyet alanını modelin anlayabilmesi için kategorig eriye çevirelim
def to_categorical(df):
    cat_columns = [
        'Gender',
    ]
    for i in cat_columns:
        df[i] = pd.Categorical(df[i])
    return df

to_categorical(cpVeriler).dtypes

#z-score kullanarak standart sapma ile aykırı verileri veri tabanından çıkaralım
def clear_outlier(df):
    for column in df.select_dtypes(exclude='category').columns:
        upper_limit = df[column].mean() + 3*df[column].std()
        lower_limit = df[column].mean() - 3*df[column].std()

        df[column] = np.where(
            df[column]>upper_limit,
            upper_limit,
            np.where(
                df[column]<lower_limit,
                lower_limit,
                df[column]
            )
        )
    return df

clear_outlier(cpVeriler)


#Kalp krizi verilerinin pozitif ve negatif sonuçlarının birbirine oranına bakalım
plt.pie(
    veriler['Result'].value_counts(),
    labels=['Positive', 'Negative'],
    explode=[0, 0.1],
    autopct='%.0f%%'
)

#Veri seti içerisinde bulunan metinsel veriyi sayısal veriye dönüştürelim
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()

result=veriler.iloc[:,-1:].values
result[:,0]=le.fit_transform(veriler.iloc[:,-1])
result=ohe.fit_transform(result).toarray()
veri=veriler.iloc[:,0:8].values


data=pd.DataFrame(data=veri,index=range(1319),columns=['Age','Gender','Heart rate','Systolic blood pressure','Diastolic blood pressure','Blood sugar','CK-MB','Troponin'])
tahmindata=pd.DataFrame(data=result[:,-1],index=range(1319),columns=['Positive'])
tumVeri=pd.concat([data,tahmindata],axis=1)


#tümverilerin bir tabloda birbirine oranına bakalım
sns.pairplot(tumVeri)
sns.heatmap(tumVeri.corr())


#Hazırladığımız veri setine bir Pipeline kuralım. Bu pipeline ile verileri veri ön işleme, modeli eğitme sonuçları görüntüleme gibi adımları içerir.
cat_cs = make_column_selector(dtype_include='category')

cat_ohe = preprocessing.OneHotEncoder(sparse_output=False, drop='if_binary').set_output(transform='pandas')
cat_pipeline = make_pipeline(cat_ohe)


num_cs = make_column_selector(dtype_exclude='category')

num_scl = preprocessing.StandardScaler().set_output(transform='pandas')
num_pipeline = make_pipeline(num_scl)

col_t = make_column_transformer(
    (num_pipeline, num_cs),
    (cat_pipeline, cat_cs)
)

preprocess = make_pipeline(
    preprocessing.FunctionTransformer(to_categorical),
    preprocessing.FunctionTransformer(clear_outlier),
    col_t
)

preprocess.set_output(transform='pandas')

#Kulanmak istediğimiz modelleri hazırlayalım
X = veriler.copy()
y = X.pop("Result")

models = {
    "AdaBoost": {
        'model': AdaBoostClassifier()
    },
    "Random Forest": {
        'model': RandomForestClassifier(verbose=False)
    },
    "QDA": {
        'model': QuadraticDiscriminantAnalysis()
    },
    "Neural Net": {
        'model': MLPClassifier(verbose=False)
    },
    "RBF SVM": {
        'model': SVC(verbose=False)
    },
    "Gaussian Process": {
        'model': GaussianProcessClassifier()
    },
    "Linear SVM": {
        'model': SVC(kernel="linear", verbose=False)
    },
    "LGBM": {
        'model': LGBMClassifier(verbose=0)
    },
    "Decision Tree": {
        'model': DecisionTreeClassifier()
    },
    "CatBoost": {
        'model': CatBoostClassifier(verbose=False)
    },
    "Naive Bayes": {
        'model': GaussianNB()
    },
    "XGB": {
        'model': XGBClassifier()
    },
    "Nearest Neighbors": {
        'model': KNeighborsClassifier()
    }
}


#Hazırladığımızı modellerin eğitim işlemini gerçekleştirelim.
for model in models:
    print('----------------------------------')
    print(f'{model} is training...')

    model_pipeline = make_pipeline(preprocess, models[model]['model'])
    scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='accuracy')
    
    models[model]['ACC'] = scores



for model in models:
    print('------------------------------------')
    print(f"{model}\nAccuracy CV 5: {models[model]['ACC']}\nAccuarcy Mean: {models[model]['ACC'].mean()}")

plot_df = pd.DataFrame({'Model': [], 'ACC': []})
for model in models:
    plot_df.loc[len(plot_df.index)] = [model, models[model]['ACC'].mean()]


fig, axs = plt.subplots(1, 1, figsize=(10, 8))

sns.barplot(x='Model', y='ACC', data=plot_df, ax=axs)
axs.set_title(f'Model Doğruluk Grafiği')
axs.set_xticklabels(axs.get_xticklabels(), rotation=45)
for p in axs.patches:
    height = p.get_height()
    width = p.get_x() + p.get_width() / 2.
    axs.text(width, height, f'{height:.2f}', ha='center', va='top', rotation=90, fontsize=8, color='white')


plt.tight_layout()
plt.show()