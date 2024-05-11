from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from flask import Flask, request
import pandas as pd

def handleNonNumericalData(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 1
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = pd.read_csv('./dataset.csv')
df = df[~df.isin(['?']).any(axis=1)]

specs = handleNonNumericalData(df[[
    "make",
    "fuel-type", 
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "num-of-cylinders",
    "horsepower",
    "peak-rpm"
]])
price = df['price']

AI = DecisionTreeClassifier()
AI = AI.fit(specs.values, price.values)

app = Flask(__name__)

@app.route("/", methods=['POST'])
def hello_world():
    global AI
    pred = 'Error on getting the price.'

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        data = np.array(list(json.values()))
        pred = AI.predict([data])
    
    return {"price": pred[0]}

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)