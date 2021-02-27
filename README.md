# Classification-Buy-SUV
This dataset includes the age and salaries of customers (features), who decide if they want to buy a SUV (dependent variable). Different classification models are used to predict the decisions of the customers. It is based on the udemy course Machine Learning A-Z. Dataset is uploaded: Social_Network_Ads.csv
* Logistic Regression Model: 201126-LogReg.ipynb
* K-NN Model: 201129-K-NN.ipynb
* linear SVM Model: 201202-linear-SVM.ipynb
* rbf kernel SVM Model: 201202-kernel-SVM.ipynb
* Naive Bayes Model: 201211_NaiveBayes.ipynb
* Decision Tree Model: 201211_DecisionTree.ipynb
* Random Forest Model: 201212_RandomForest.ipynb

## Expose the Random Forest ML Service as an API
* prepare and train the model, save the model in the file 'rf.pkl': prepare_model.py
* use flask to create a web API, autogenerate a user interface with flasgger: flask_predict_api.py
