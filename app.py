import pandas as pd
from flask import Flask, render_template,request
import re
from scipy.stats import zscore
from scipy.stats import ttest_ind
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix,classification_report
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import sys
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os


app = Flask(__name__)

# Load data
df = pd.read_excel('Telco_customer_churn.xlsx')
work_df = df.copy()

# Data preprocessing functions
# Formating _columns names into snake case:
def camel_to_snake(name):
    name = name.replace(' ', '_')
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1\2', s1).lower()
# Converting categorical values to numerical: 
def convert_cat_num(work_df):
    work_df['gender'] = work_df["gender"].map({'Male':0,'Female':1})
    work_df['senior_citizen'] = work_df["senior_citizen"].map({'Yes':1,'No':0})
    work_df['partner'] = work_df['partner'].map({'Yes':1,'No':0})
    work_df['dependents'] = work_df.dependents.map({'Yes':1,'No':0})
    work_df['phone_service'] = work_df["phone_service"].map({'Yes':1,'No':0})
    work_df['multiple_lines'] = work_df["multiple_lines"].map({'Yes':1,'No':0,'No phone service':0})
    work_df["internet_service"] = work_df["internet_service"].map({'DSL':1,'Fiber optic':1,'No':0})
    work_df['online_security'] = work_df["online_security"].map({'Yes':1,'No':0,'No internet service':0})
    work_df['online_backup'] = work_df["online_backup"].map({'Yes':1,'No':0,'No internet service':0})
    work_df['paperless_billing'] = work_df["paperless_billing"].map({'Yes':1,'No':0})
    work_df['tech_support'] = work_df["tech_support"].map({'Yes':1,'No':0,'No internet service':0})
    work_df['streaming_tv'] = work_df["streaming_tv"].map({'Yes':1,'No':0,'No internet service':0})
    work_df['streaming_movies'] = work_df["streaming_movies"].map({'Yes':1,'No':0,'No internet service':0})
    work_df['device_protection'] = work_df["device_protection"].map({'Yes':1,'No':0,'No internet service':0})
    work_df["contract"] = work_df["contract"].map({'Month-to-month':0, 'Two year':2, 'One year':1})
    work_df["payment_method"] = work_df["payment_method"].map({'Electronic check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 0, 'Mailed check': 3})
    work_df['churn_label'] = work_df["churn_label"].map({'Yes':1,'No':0})
    return work_df

def remove_nulls(work_df):
    K = work_df.isnull().sum()
    for i in K:
        if i > 0 :
            ind = K[K == i].index[0]
            if i/work_df.shape[0] < 0.4:
                work_df = work_df.dropna(subset=[ind])
            else:
                work_df = work_df.drop([ind], axis = 1)
    return work_df

def preprocess_data(work_df):
    # Column name Case Conversion
    camel_case_columns = work_df.columns
    snake_case_columns = [camel_to_snake(col) for col in camel_case_columns]
    work_df.columns = snake_case_columns
    # Consistency in data
    if 'latitude' in work_df.columns :
        work_df = work_df[(work_df['latitude'] >= -90) | (work_df['latitude'] <= 90)]
        work_df = work_df[(work_df['longitude'] >= -180) | (work_df['longitude'] <= 180)]
    # Formatting data
    work_df['total_charges'] = pd.to_numeric(work_df['total_charges'], errors='coerce')
    # Drop/Remove missing values:
    print('before',work_df.shape)
    work_df = remove_nulls(work_df)
    print('after',work_df.shape)
    # Dropping unwanted columns
    if 'lat_long' in work_df.columns :
        work_df = work_df.drop('lat_long',axis = 1)
    # Removing Duplicates:
    work_df = work_df.drop_duplicates(keep='first')
    # Converting Categorical to Numerical:
    convert_cat_num(work_df)
    # Feature Reduction:
    if 'count' in work_df.columns:
        work_df = work_df.drop(['count'], axis=1)
    if 'country' in work_df.columns:
        work_df = work_df.drop(['country'], axis=1)
    if 'state' in work_df.columns:
        work_df = work_df.drop(['state'], axis=1)
    # Resetting Index
    df = work_df.reset_index(drop=True)
    return df



def model(method, x_train, y_train, x_test, y_test,md):
    # Training
    method.fit(x_train, y_train)
    # Predictions
    predictions = method.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    c_matrix = confusion_matrix(y_test, predictions)
    percentages = (c_matrix / np.sum(c_matrix, axis=1)[:, np.newaxis]).round(2) * 100
    labels = [[f"{c_matrix[i, j]} ({percentages[i, j]:.2f}%)" for j in range(c_matrix.shape[1])] for i in range(c_matrix.shape[0])]
    labels = np.asarray(labels)
    # Plot
    # sns.heatmap(c_matrix, annot=labels, fmt='', cmap='Blues')
    # plt.title('Confusion Matrix')
    # Evaluate model performance
    print("RMSE:", rmse)
    print("ROC AUC: ", '{:.2%}'.format(roc_auc_score(y_test, predictions)))
    print("Model accuracy: ", '{:.2%}'.format(accuracy_score(y_test, predictions)))
    print(f'Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions)}')
    print(classification_report(y_test, predictions))
    if md == 'lr':
      coefficients = lr.coef_
      feature_names = feature_data.columns
      feature_coefficients = list(zip(feature_names, coefficients[0]))
      feature_coefficients = sorted(feature_coefficients, key=lambda x: abs(x[1]), reverse=True)
      for feature, coefficient in feature_coefficients:
          print(f'{feature}: {coefficient:.4f}')
    return method


df_edited = preprocess_data(work_df)
xgb = XGBClassifier(learning_rate= 0.01,max_depth = 6,n_estimators = 1000)
features = ['gender','senior_citizen', 'tenure_months',
'phone_service', 'multiple_lines', 'internet_service','online_security',
             'online_backup', 'tech_support',
'streaming_tv', 'streaming_movies', 'monthly_charges', 'total_charges']
feature_data = df_edited[features]
x = feature_data.values
y = df_edited['churn_label'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state =2, test_size = 0.2)
xgb_t = model(xgb,x_train,y_train,x_test,y_test,'xgb')
feature_importance = xgb_t.feature_importances_
feature_names = feature_data.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
#Plot
# plt.figure(figsize=(10, 6))
# plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('Feature Importance')
# plt.show()



@app.route('/start', methods=['GET', 'POST'])
def index():
    # return render_template('web_page_first_copy.html', df=df_edited)
    return render_template('start.html')

@app.route('/inp_data', methods=['GET', 'POST'])
def inp():
    if request.method == 'POST':
        print(request)
        file = request.files['upload-file']
        print(file)
        uploads_folder = os.path.join(os.getcwd(), 'uploads')
        path = os.path.join(os.getcwd(), 'uploads') + file.filename
        print(path)
        file.save(path)
        input_data = pd.read_excel(path)
        inp_2 = preprocess_data(input_data)
        feature_data_inp = inp_2[features]
        predictions = pd.DataFrame(inp_2['customerid'])
        predictions['churn_label'] = pd.DataFrame(xgb_t.predict(feature_data_inp))
        predictions['churn'] = predictions['churn_label'].apply(lambda x: 'Yes' if x == 1 else 'No')
        f_s_1  = f'Total number of customers predicted:{predictions.shape[0]}'
        churn = (predictions[predictions['churn_label']==1].shape[0]) / predictions.shape[0]
        f_s_2  = f'Percentages of customers churned :{churn*100 }'
        f_s = f_s_1+'\n'+f_s_2
        print(f_s)
        output_strategy_1 = '''Offering personalized retention offers, discounts, or incentives to customers identified as
high-risk by XGB and RF.'''
        output_strategy_2 = '''Implementing proactive customer support measures to address high-risk customers'
concerns and issues promptly'''
        return render_template('data.html', data=predictions.to_html(),f_s_1 = f_s_1, f_s_2= f_s_2, out_s_1 = output_strategy_1,out_s_2 = output_strategy_2)

@app.route('/input_data_format', methods=['GET', 'POST'])
def data_format():
    return render_template('input_data_format.html')


if __name__ == '__main__':
    app.run(debug=True)





