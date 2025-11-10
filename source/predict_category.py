import pandas as pd
import joblib
import re

#load the model
model = joblib.load("model/product_classifier_model.pkl")
print("Model loaded successfully")
print("Type exit at any given momment to exit the program")

while True:
    title = input("Enter product title:")
    if title.lower() =="exit":
        print("Exiting ....")
        break
    
    is_frd_freezer = int("serie 4 kgv39vl31g" in title.lower() or "sbs8004po" in title.lower())
    user_input = pd.DataFrame([
        {"Product Title":title,
        "is Fridge Freezers": is_frd_freezer 
         }
    ])
    
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}")