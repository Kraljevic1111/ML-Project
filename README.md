# ML-Project

This project uses machine learning to automatically classify products based on their titles. It combines text features with custom signals to improve prediction accuracy. The model is trained using a pipeline that includes text vectorization and scaling, and it’s saved for interactive prediction.


Project Structure

ml-project/
├── data/                  # Contains the training dataset (CSV)
├── model/                 # Stores the trained model (.pkl)
├── Notebook/              # Jupyter notebooks for exploration and experimentation
├── source/                # Python scripts for training and prediction
├── .gitignore             # Prevents tracking of model files and folders
└── README.md              # Project overview and instructions

🏋️‍♂️ Model Training
The training script () performs the following steps:
• 	Loads product data from 
• 	Cleans and preprocesses the text and labels
• 	Drops irrelevant columns
• 	Adds a binary signal () based on keyword presence
• 	Transforms features using  and 
• 	Trains an SVM classifier using a pipeline
• 	Saves the trained model to 
To train the model:

source/train_model.py

Prediction Script
The prediction script () allows interactive testing:
• 	Loads the trained model
• 	Prompts the user to enter a product title
• 	Adds the same binary signal () based on keywords
• 	Predicts the product category
• 	Continues until the user types 
To run the prediction:

source/predict_category.py

Git Ignore Setup

Inoring the model folder and files with extension .pkl
model/
*.pkl


