#importing needed librarys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib

# Load dataset
df = pd.read_csv("data/products.csv")

#removing missing values 

df = df.dropna()

#cleaning the columns which wee are going to use for training

df['Product Title'] = df['Product Title'].astype(str).str.lower().str.strip()
df[' Category Label'] = df[' Category Label'].astype(str).str.lower()

#removing columns that are not useful for training

df = df.drop(columns=['product ID', 'Merchant ID', '_Product Code', 'Number_of_Views', 'Merchant Rating', ' Listing Date  '])

# Creating a new column to improve accuracy of model

df['is Fridge Freezers'] = df['Product Title'].astype(str).str.lower().str.strip().str.contains("serie 4 kgv39vl31g","sbs8004po").astype(int)



#faetures and labels

x = df[["Product Title","is Fridge Freezers"]]
y = df[" Category Label"]


#transforming data 
preprocesor = ColumnTransformer([

    ("title",TfidfVectorizer(),"Product Title"),
    ("checking",MinMaxScaler(),['is Fridge Freezers'])
])


#creating a pipeline

pipeline= Pipeline([
       ("preprocesing", preprocesor),
       ("Classifier", SVC())])

#training the models
pipeline.fit(x,y)

joblib.dump(pipeline,"model/product_classifier_model.pkl")

print("Model trained and saved")