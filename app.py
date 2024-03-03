from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
import plotly.express as px
from datetime import datetime
from pymongo import MongoClient
from prophet import Prophet
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
import numpy as np

app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb+srv://nivedha:nivedhamongodb@cluster0.h0jt46s.mongodb.net/entries'
mongo = PyMongo(app)

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
        'date': datetime.utcnow()
    }

    mongo.db.incomeandexpense.insert_one(entry)
    return redirect(url_for('index'))

@app.route('/predict_expense')
def predict_expense():
    # Implement prediction logic based on historical data
    # This could involve training a machine learning model on past entries
    # For simplicity, we'll assume a basic rule-based prediction for now
    
    mongo_uri = "mongodb+srv://nivedha:nivedhamongodb@cluster0.h0jt46s.mongodb.net/"
    client = MongoClient(mongo_uri)

    db = client.income_expense_data
    collection = db.transactions

    transaction = list(collection.find()) 
    transaction_df = pd.DataFrame(transaction)
    
    transaction_df['Amount']=transaction_df['Amount'].astype('float')
    missing_values=transaction_df.columns[transaction_df.isna().any()]
    transaction_df=transaction_df.drop(['Account'],axis=1)

    income_df=transaction_df[transaction_df['Income/Expense']== 'Income']
    income_df['Amount']=income_df['Amount'].astype('float')
    
    expense_df=transaction_df[transaction_df['Income/Expense']== 'Expense']
    expense_df['Amount']=expense_df['Amount'].astype('float')
    
    categories = expense_df['Category']
    category_counts = categories.value_counts()
    #most spent category
    most_frequent_category = category_counts.idxmax()
    #average spending per category
    #ERROR avg_spending_category=expense_df.groupby('Category').mean().sort_values(by='Amount')
    #total expense
    expense_amount=expense_df.Amount.sum()
    #total income
    income_amount=income_df.Amount.sum()

    expense_df['Date']=pd.to_datetime(expense_df.Date)
    income_df['Date']=pd.to_datetime(income_df.Date)
    expense_df['Date'] = pd.to_datetime(expense_df['Date'], format='%d-%M-%Y')
    income_df['Date'] = pd.to_datetime(income_df['Date'], format='%d-%M-%Y')

    date=expense_df['Date'].dt.month
    expense_df.set_index('Date').groupby(pd.Grouper(freq='M'))

    #monthly spending
    monthly_spending = expense_df.set_index('Date').groupby(pd.Grouper(freq='M')).Amount
    monthly_income = income_df.set_index('Date').groupby(pd.Grouper(freq='M')).Amount

    
    #Prophet model ----- start
    category_code={'Food':0, 'Other':1, 'Transportation':2, 'Apparel':3, 'Household':4,
       'Social Life':5, 'Education':6, 'Self-development':7, 'Beauty':8, 'Gift':9}
    expense_df['category_code']=expense_df.Category.map(category_code)
    
    expense_df_TSA=expense_df
    expense_df_TSA=expense_df_TSA.drop(['Category','Income/Expense','category_code'],axis=1)
    expense_df_TSA.rename(columns={'Date':'ds','Amount':'y'},inplace=True)
    p=Prophet(interval_width=0.92,daily_seasonality=True)
    model= p.fit(expense_df_TSA)
    future = p.make_future_dataframe(periods=36,freq='M')
    forecast_prediction = p.predict(future)
    next_month_prediction = forecast_prediction[forecast_prediction['ds'] == forecast_prediction['ds'].max()]['yhat'].values[0]
    #Prophet model ----- end


    predicted_expense = next_month_prediction  # Replace with your prediction logic
    return render_template('predict_expense.html', predicted_expense=predicted_expense)

@app.route('/dashboard')
def dashboard():
    entries = list(mongo.db.entries.find())
    df = pd.DataFrame(entries)
    df['date'] = pd.to_datetime(df['date'],format='%d-%M-%Y')
    fig = px.bar(df, x='date', y='amount', color='category', title='Income/Expense Overview')

    graphJSON = px.to_json(fig)
    return render_template('dashboard.html', graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
