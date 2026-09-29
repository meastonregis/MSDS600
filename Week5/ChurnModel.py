# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class ChurnModel:
    """
    Modified functions from Week 5 FTE, expanded to be a class and include additional
    information.
    Load Data, Make a Prediction (Pre-Processed), Present Prediction and
    probabilities.
    Defaults to Pre-Proccessed Data (PPD) = True. Currently only works with
    Pre-Processed Data.
    """
    
    def __init__(self, filepath, bestmodel, PPD = True, last_prediction = None):
        self.filepath = filepath
        self.bestmodel = bestmodel
        self.PPD = PPD
        self.last_prediction = last_prediction

    def develop_predictions(self):
        import pandas as pd
        from pycaret.classification import predict_model, load_model
        
        def load_data(filepath):
            """
            Loads Churn data into a DataFrame from a string filepath.
            """
            df = pd.read_csv(filepath, index_col='customerID')
            return df
        
        def make_predictions(self, mpdf):
            """
            Uses the pycaret best model to make predictions on data in the df dataframe.
            """
            model = load_model(self.bestmodel)
            mppredictions = predict_model(model, data=mpdf)

            # Check the column names
            print(mppredictions.columns)
            
            # Rename 'prediction_label' to 'Churn_prediction' if it exists
            if 'prediction_label' in mppredictions.columns:
                mppredictions.rename(columns={'prediction_label': 'Churn_Prediction', 'prediction_score':'Churn_Probability'}, inplace=True)
                
                # Replace values in the new column
                mppredictions['Churn_Prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
                
                #Add a new Column with the Prediction Percentages
                if self.PPD == True:                
                    return mppredictions.drop(['tenure','PhoneService','Contract','PaymentMethod','MonthlyCharges','TotalCharges','charge_per_tenure'], axis=1)
                else:
                    return mppredictions.drop(['tenure','PhoneService','Contract','PaymentMethod','MonthlyCharges','TotalCharges'], axis=1)
            else:
                raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")

        
        df = load_data(self.filepath)
        self.last_prediction = make_predictions(self,df)
    
    def show_prediction(self):
        print("Prediction:")
        print(self.last_prediction)