from flask import Flask, redirect, url_for, render_template, request
from test import apply

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict/", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        text = request.form["Text"]
#        print(apply(text))
        return render_template("index.html", res=apply(text))
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)