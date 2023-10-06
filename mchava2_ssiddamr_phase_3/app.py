from flask import Flask, render_template,request
from model import xgbModel
import pandas as pd

app = Flask(__name__)

@app.route("/airbnb")
def home_page():
    return render_template("home_page.html")

@app.route("/",methods=["POST"])
def get_data_from_home():
    I1=request.form['gender']
    I2=request.form['age']
    I3=request.form['signup']
    I4=request.form['language']
    I5=request.form['marketing_channel']
    I6=request.form['marketing_provider']
    I7=request.form['application']
    I8=request.form['device']
    I9=request.form['browser']
    I10=request.form['year']

    user_data = pd.DataFrame([[I1,I2,I3,I4,I5,I6,I7,I8,I9,I10]])

    prediction = xgbModel.predict(user_data)

    return render_template('destination_prediction.html',pred=prediction)











if __name__ == "__main__":
    app.run(debug=True)



