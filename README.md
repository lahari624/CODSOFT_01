🚢 Titanic Survival Prediction with Random Forest

A machine learning project that predicts whether a passenger survived the Titanic disaster using classification algorithms—primarily the Random Forest Classifier.


🎯 Project Objective

The goal is to use machine learning to build a predictive model that can determine whether a Titanic passenger would survive based on historical data.

📊 Dataset Overview :
 
Source: Kaggle Titanic Dataset

Target Variable: Survived (0 = No, 1 = Yes)

Features Used:

Pclass – Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

Sex – Gender (0 = Female, 1 = Male)

Age – Age in years

SibSp – # of siblings/spouses aboard


🧹 Data Preprocessing : 

Dropped irrelevant columns: PassengerId, Name, Ticket, Cabin

Handled missing values:

Filled Age with median

Filled Embarked with mode

Encoded categorical features using LabelEncoder:

Sex: Female → 0, Male → 1

Embarked: C → 0, Q → 1, S → 2

Parch – # of parents/children aboard

Fare – Passenger fare

Embarked – Port of Embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)


🧠 Model Training : 

Algorithm Used: RandomForestClassifier

Steps:

  1) Split data into train/test sets (80/20)

  2) Trained the classifier using training data

  3) Saved the trained model using joblib


📈 Model Evaluation : 

Metrics:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Results were visualized and printed to console


🧾 Project Structure : 

Titanic-Prediction

├── titanic_model.pkl           
├── Titanic-Dataset.csv         
├── titanic_prediction.ipynb     
├── README.md                     
└── requirements.txt             

⚙️ Requirements : 

Install the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn joblib


🚀 How to Run : 

  1) Clone the repo or download the files

  2) Ensure you have Python 3 installed

  3) Run the Jupyter notebook or script

  4) Use the saved model (titanic_model.pkl) for future predictions


📌 Future Improvements :

Use GridSearchCV or RandomizedSearchCV for hyperparameter tuning

Apply more advanced feature engineering

Try other models like XGBoost or SVM

Build a simple Streamlit web app for interactive prediction


