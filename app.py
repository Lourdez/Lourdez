from flask import Flask,render_template,url_for,request
from flask_material import Material
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 

import sklearn.externals
import joblib


app = Flask(__name__)
Material(app)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/analyze_form')
def analyze_form():
    return render_template("analyze_form.html")

@app.route('/analyze',methods=["POST"])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps'] 
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        
        model_choice = request.form['model_choice']
        
		# Clean the data by convert from unicode to float 
        sample_data = [age,sex,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,ca,cp,cp,cp,cp,thal,thal,thal,thal,slope,slope,slope]
        sample_data1 = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        clean_data = [float(i) for i in sample_data]
        print(sample_data)
        print(sample_data1)

		# Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model_choice == 'rfmodel':
            rf_model = joblib.load('data/random_forest.pkl')
            result_prediction = rf_model.predict(ex1)
         
        elif model_choice == 'nbmodel':
            nb_model = joblib.load('data/naive_bayes_model.pkl')
            result_prediction = nb_model.predict(ex1)
        elif model_choice == 'lrmodel':
            lr_model = joblib.load('data/logistic_regression.pkl')
            result_prediction = lr_model.predict(ex1)
        elif model_choice == 'dtmodel': 
            dt_model = joblib.load('data/decision_tree.pkl')
            result_prediction = dt_model.predict(ex1)
        
        
        methods = ["Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest"]
        accuracy = [86.89, 78.69, 86.89, 88.89]
        colors = ["b", "c","m","y"]
        sns.set_style("whitegrid")
        plt.figure(figsize=(16,5))
        plt.yticks(np.arange(0,100,10))
        plt.ylabel("Accuracy %")
        plt.xlabel("Algorithms")
        sns.barplot(x=methods, y=accuracy, palette=colors)
        plt.savefig('static/images/comparison.png')


    return render_template('result.html',
                           age = age, sex = sex,cp = cp,trestbps = trestbps,
                           chol = chol,fbs = fbs,restecg = restecg,thalach = thalach,
                           exang = exang,oldpeak = oldpeak,slope = slope,ca = ca,
                           thal = thal,sample_data1=sample_data1,
                           result_prediction=result_prediction,model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
