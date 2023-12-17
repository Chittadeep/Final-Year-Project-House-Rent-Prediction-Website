import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model1.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='Rent should be Rs. {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
