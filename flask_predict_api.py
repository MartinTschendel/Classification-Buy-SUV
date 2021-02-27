import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

with open('rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """Example endpoint returning a prediction of buyers decision (1 = purchase; 0 = no purchase)
    ---
    parameters:
      - name: Age
        in: query
        type: number
        required: true
      - name: EstimatedSalary
        in: query
        type: number
        required: true
    """
    Age = request.args.get("Age")
    EstimatedSalary = request.args.get("EstimatedSalary")
    
    prediction = model.predict(np.array([[Age, EstimatedSalary]]))
    return str(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#type in browser: localhost:5000/apidocs/
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    