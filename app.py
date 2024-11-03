from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import xgboost as xgb

global filename
global df, X_train, X_test, y_train, y_test
global lgb_model

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = df[["sbp","tobacco","ldl","adiposity","famhist","typea","obesity","alcohol","age"]]
    y = np.array(df["chd"])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")

def adaboost():
    global ada_acc
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for AdaBoost is {ada_acc * 100}%\n'
    text.insert(END, result_text)

def decision_tree():
    global dt_acc
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Decision Tree is {dt_acc * 100}%\n'
    text.insert(END, result_text)

def gradient_boosting():
    global gb_acc
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Gradient Boosting Machine is {gb_acc * 100}%\n'
    text.insert(END, result_text)

def bagging():
    global bg_acc
    bg = BaggingClassifier(n_estimators=100, random_state=0)
    bg.fit(X_train, y_train)
    y_pred = bg.predict(X_test)
    bg_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Bagging Classifier is {bg_acc * 100}%\n'
    text.insert(END, result_text)

def xgboost():
    global xgb_acc
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'max_depth': 3, 'eta': 1, 'objective': 'multi:softmax', 'num_class': len(np.unique(y_train))}
    xgb_model = xgb.train(params, dtrain)
    y_pred = xgb_model.predict(dtest)
    xgb_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for XGBoost is {xgb_acc * 100}%\n'
    text.insert(END, result_text)

def lightgbm():
    global lgb_acc, lgb_train, lgb_model
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': len(np.unique(y_train))}
    lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_test)
    y_pred = np.argmax(lgb_model.predict(X_test), axis=1)
    lgb_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for LightGBM is {lgb_acc * 100}%\n'
    text.insert(END, result_text)

def plot_bar_graph():
    algorithms = ['AdaBoost', 'Decision Tree', 'Gradient Boosting', 'Bagging', 'XGBoost', 'LightGBM']
    accuracies = [ada_acc * 100, dt_acc * 100, gb_acc * 100, bg_acc * 100, xgb_acc * 100, lgb_acc * 100]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']
    
    plt.bar(algorithms, accuracies, color=colors)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Machine Learning Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict():
    global lgb_model
    # Open file manager to select CSV file
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if filename:
        # Read the selected CSV file
        input_data = pd.read_csv(filename)

        # Fill missing values with mode for each column
        input_data.fillna(input_data.mode().iloc[0], inplace=True)

        # Preprocess input data (if needed)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])

        # Perform prediction using LightGBM model
        y_pred = np.argmax(lgb_model.predict(input_data), axis=1)

        # Display the prediction result
        if y_pred[0] == 1:
            text.insert(END,"Coronary Heart Disease Detected" )

            messagebox.showinfo("Prediction Result", "Coronary Heart Disease Detected")
        else:
            text.insert(END,"Coronary Heart Disease  not Detected" )
            messagebox.showinfo("Prediction Result", "Coronary Heart Disease Not Detected")


main = tk.Tk()
main.title("Predicting Coronary heart disease using an improved LightGBM model") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = tk.Label(main, text='Predicting Coronary heart disease using an improved LightGBM model',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=600)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=250, y=600)

splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, bg="light green", width=15)
splitButton.place(x=450, y=600)
splitButton.config(font=font1)

adaboostButton = tk.Button(main, text="AdaBoost", command=adaboost, bg="lightgrey", width=15)
adaboostButton.place(x=50, y=650)
adaboostButton.config(font=font1)

dtButton = tk.Button(main, text="Decision Tree", command=decision_tree, bg="pink", width=15)
dtButton.place(x=250, y=650)
dtButton.config(font=font1)

gbButton = tk.Button(main, text="Gradient Boosting", command=gradient_boosting, bg="yellow", width=15)
gbButton.place(x=450, y=650)
gbButton.config(font=font1)

baggingButton = tk.Button(main, text="Bagging", command=bagging, bg="lightgreen", width=15)
baggingButton.place(x=650, y=650)
baggingButton.config(font=font1)

xgbButton = tk.Button(main, text="XGBoost", command=xgboost, bg="lightblue", width=15)
xgbButton.place(x=850, y=650)
xgbButton.config(font=font1)

lgbButton = tk.Button(main, text="LightGBM", command=lightgbm, bg="orange", width=15)
lgbButton.place(x=1050, y=650)
lgbButton.config(font=font1)

plotButton = tk.Button(main, text="Plot Results", command=plot_bar_graph, bg="lightgrey", width=15)
plotButton.place(x=50, y=700)
plotButton.config(font=font1)

predict_button = tk.Button(main, text="Prediction", command=predict, bg="orange", width=15)
predict_button.place(x=250, y=700)
predict_button.config(font=font1)

main.config(bg='#32d1a7')
main.mainloop()
