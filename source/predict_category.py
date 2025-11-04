import pandas as pd
import joblib

#load the model
model = joblib.load("model/product_classifier_model.pkl")
print("Model loaded successfully")
print("Type exit at any given momment to exit the program")

while True:
    title = input("Enter product title:")
    if title.lower() =="exit":
        print("Exiting ....")
        break
    
    user_input = pd.DataFrame([
        {"combined title":title}
    ])
    
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}")