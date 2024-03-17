from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
#from plotly import express as px
from datetime import datetime
from pymongo import MongoClient
from prophet import Prophet
import pandas as pd
#import plotly.io as pio

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
import numpy as np



app = Flask(__name__)


app.config['MONGO_URI'] = 'mongodb+srv://nivethaa0310:nivethaa@cluster0.wp2w8d4.mongodb.net/Transaction'
mongo = PyMongo(app)

mongo_uri = "mongodb+srv://nivethaa0310:nivethaa@cluster0.wp2w8d4.mongodb.net/"
client = MongoClient(mongo_uri)

db = client.Transaction
collection = db.transactions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_entry', methods=['POST'])
def add_entry():
    amount = float(request.form['amount'])
    entry_type = request.form['type']
    category = request.form['category']
    note = request.form['note']

    entry = {
        'amount': amount,
        'type': entry_type,
        'category': category,
        'note': note,
        'date': datetime.utcnow().strftime('%m-%d-%y %H:%M')
    }

    mongo.db.transactions.insert_one(entry)
    return redirect(url_for('index'))

@app.route('/predict_expense')
def predict_expense():
    # Implement prediction logic based on historical data
    # This could involve training a machine learning model on past entries
    # For simplicity, we'll assume a basic rule-based prediction for now
    
    transaction = list(collection.find()) 
    transaction_df = pd.DataFrame(transaction)
    
    transaction_df['amount']=transaction_df['amount'].astype('float')
    missing_values=transaction_df.columns[transaction_df.isna().any()]

    income_df=transaction_df[transaction_df['type']== 'Income']
    income_df['amount']=income_df['amount'].astype('float')
    
    expense_df=transaction_df[transaction_df['type']== 'Expense']
    expense_df['amount']=expense_df['amount'].astype('float')
    
    categories = expense_df['category']
    category_counts = categories.value_counts()
    #most spent category
    most_frequent_category = category_counts.idxmax()
    #average spending per category
    avg_spending_category = expense_df.groupby('category').agg({'amount': 'mean'}).reset_index()
    #total expense
    expense_amount=expense_df.amount.sum()
    #print("expense amount: "+expense_amount)
    #total income
    income_amount=income_df.amount.sum()
    #print("income amount: "+income_amount)

    expense_df['date'] = pd.to_datetime(expense_df['date'], errors='coerce')
    income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')

    # Drop rows with missing or incorrect dates
    expense_df = expense_df.dropna(subset=['date'])
    income_df = income_df.dropna(subset=['date'])

    date=expense_df['date'].dt.month
    expense_df.set_index('date').groupby(pd.Grouper(freq='M'))

    #monthly spending
    monthly_spending = expense_df.set_index('date').groupby(pd.Grouper(freq='M')).amount
    monthly_income = income_df.set_index('date').groupby(pd.Grouper(freq='M')).amount
    #print("monthly spending: "+monthly_spending)
    #print("monthly income: "+monthly_income)
    
    #Prophet model ----- start
    category_code={'Bank':0, 'Clothing':1, 'Debt':2, 'Donations':3, 'Education':4,
       'Entertainment':5, 'Fee':6, 'Food':7, 'Gift':8, 'Groceries':9, 'Healthcare':10, 'Home':11, 
       'Insurance':12, 'Payroll':13, 'Personal':14, 'Savings':15, 'Subscription':16,
       'Tax':17, 'Transportation':18, 'Utilities':19}
    expense_df['category_code']=expense_df.category.map(category_code)
    
    expense_df_TSA=expense_df
    expense_df_TSA=expense_df_TSA.drop(['category','type','category_code'],axis=1)
    expense_df_TSA.rename(columns={'date':'ds','amount':'y'},inplace=True)
    p=Prophet(interval_width=0.92,daily_seasonality=True)
    model= p.fit(expense_df_TSA)
    future = p.make_future_dataframe(periods=36,freq='M')
    forecast_prediction = p.predict(future)
    next_month_prediction = forecast_prediction[forecast_prediction['ds'] == forecast_prediction['ds'].max()]['yhat'].values[0]
    #Prophet model ----- end

    predicted_expense = next_month_prediction  # Replace with your prediction logic
    return render_template('predict_expense.html', predicted_expense=predicted_expense,avg_spending_category=avg_spending_category)

# @app.route('/dashboard')
# def dashboard():
#     mongo_uri = "mongodb+srv://nivedha:nivedhamongodb@cluster0.h0jt46s.mongodb.net/"
#     client = MongoClient(mongo_uri)

#     db = client.income_expense_data
#     collection = db.transactions

#     transaction = list(collection.find()) 
#     df = pd.DataFrame(transaction)

#     df['Date'] = pd.to_datetime(df['Date'])
#     fig = px.bar(df, x='Date', y='Amount', color='Category', title='Income/Expense Overview')

#     graphJSON = pio.to_json(fig)
#     return render_template('dashboard.html', graphJSON=graphJSON)

@app.route('/get_entries', methods=['GET'])
def get_entries():
    entries = list(mongo.db.transactions.find())
    return jsonify(entries)

if __name__ == '__main__':
    app.run(debug=True)
