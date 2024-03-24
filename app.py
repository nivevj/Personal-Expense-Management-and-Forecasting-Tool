from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_pymongo import PyMongo
from datetime import datetime
from pymongo import MongoClient
from prophet import Prophet
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure
import numpy as np

app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb+srv://nivedha:nivedhamongodb@cluster0.h0jt46s.mongodb.net/expenseprediction'
mongo = PyMongo(app)

mongo_uri = "mongodb+srv://nivedha:nivedhamongodb@cluster0.h0jt46s.mongodb.net"
client = MongoClient(mongo_uri)
db = client.expenseprediction
collection = db.transactions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
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
    most_frequent_category = category_counts.idxmax()

    avg_spending_category = expense_df.groupby('category').agg({'amount': 'mean'}).reset_index()
    avg_spending_category['amount'] = avg_spending_category['amount'].round().astype(int)
    
    expense_amount=expense_df.amount.sum()
    income_amount=income_df.amount.sum()
    expense_df['date'] = pd.to_datetime(expense_df['date'], errors='coerce')
    income_df['date'] = pd.to_datetime(income_df['date'], errors='coerce')
    expense_df = expense_df.dropna(subset=['date'])
    income_df = income_df.dropna(subset=['date'])
    date=expense_df['date'].dt.month
    expense_df.set_index('date').groupby(pd.Grouper(freq='M'))
    monthly_spending = expense_df.set_index('date').groupby(pd.Grouper(freq='M')).amount
    monthly_income = income_df.set_index('date').groupby(pd.Grouper(freq='M')).amount
    
    # Prophet model
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
    next_month_prediction=next_month_prediction.round().astype(int)
    predicted_expense = next_month_prediction
    return render_template('predict_expense.html', predicted_expense=predicted_expense,avg_spending_category=avg_spending_category) 

@app.route('/transactions_this_month')
def transactions_this_month():
    
    # total dataset
    transactions = list(mongo.db.transactions.find({
        '$or': [{'type': 'Expense'}, {'type': 'Income'}]
    }).sort([('date', -1)]))
    transaction = list(collection.find()) 
    transaction_df = pd.DataFrame(transaction)
    transaction_df['amount']=transaction_df['amount'].astype('float')
    missing_values=transaction_df.columns[transaction_df.isna().any()]
    income_df=transaction_df[transaction_df['type']== 'Income']
    income_df['amount']=income_df['amount'].astype('int')
    expense_df=transaction_df[transaction_df['type']== 'Expense']
    expense_df['amount']=expense_df['amount'].astype('int')
    expense_amount=expense_df.amount.sum()
    income_amount=income_df.amount.sum()
    total_balance=income_amount-expense_amount 

    # for current month
    transactions_month = list(mongo.db.transactions.find({
    '$or': [{'type': 'Expense'}, {'type': 'Income'}],
    'date': {'$regex': '^2'}}).sort([('date', -1)]))

    transaction_month = list(collection.find()) 
    transaction_df_month = pd.DataFrame(transactions_month)
    transaction_df_month['amount']=transaction_df_month['amount'].astype('float')
    missing_values_month=transaction_df_month.columns[transaction_df.isna().any()]
    income_df_month=transaction_df_month[transaction_df_month['type']== 'Income']
    income_df_month['amount']=income_df_month['amount'].astype('int')
    expense_df_month=transaction_df_month[transaction_df_month['type']== 'Expense']
    expense_df_month['amount']=expense_df_month['amount'].astype('int')
    expense_amount_month=expense_df_month.amount.sum()
    income_amount_month=income_df_month.amount.sum()
    total_balance_month=income_amount_month-expense_amount_month 

    predicted_month_expense = 80

    return render_template('transactions_month.html',transactions_month=transactions_month, 
    transactions=transactions,expense_amount=expense_amount,income_amount=income_amount,
    total_balance=total_balance,total_balance_month=total_balance_month,income_amount_month=income_amount_month,
    expense_amount_month=expense_amount_month, predicted_month_expense=predicted_month_expense)

@app.route('/calculator')
def calc():
    return render_template('calculator.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html', graph='plot.png')

@app.route('/plot.png')
def plot_png():
    fig = Figure(figsize=(10, 7))  # Adjust figure size if needed
    fig.subplots_adjust(bottom=0.15)  # Set bottom spacing to a smaller value

    # Add subplot for income alone
    axis1 = fig.add_subplot(1, 1, 1)  # Use a single subplot
    axis1.set_title('Income')


    data = list(collection.find({}))

    # Convert 'date' field to datetime
    for entry in data:
        entry['date'] = pd.to_datetime(entry['date'])

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-31')

    # Filter data between start and end dates
    filtered_data = [entry for entry in data if start_date <= entry['date'] <= end_date]

    date = [entry['date'] for entry in filtered_data]
    income = []
    expense = []

    for entry in filtered_data:
        if entry['type'] == 'Income':
            income.append(entry['amount'])
            expense.append(0)  # Replace None with 0 for expense
        else:
            income.append(0)  # Replace None with 0 for income
            expense.append(entry['amount'])

    # Plot income alone
    axis1.bar(date, income, color='green', label='Income')
    #axis1.fill_between(date, np.nan_to_num(income), color='lightgreen')
    axis1.set_xlabel('Date')
    axis1.set_ylabel('Income ($)')
    axis1.legend() 

    output = BytesIO()
    fig.savefig(output, format='png', bbox_inches='tight')
    output.seek(0)

    return send_file(output, mimetype='image/png')

# Expense 
    
@app.route('/plot1.png')
def plot1_png():
    fig = Figure(figsize=(10,6)) 
    fig.subplots_adjust(hspace=0.5)

    # Add subplot for income alone
    axis2 = fig.add_subplot(1, 1, 1)  # 3 rows, 1 column, subplot 1
    axis2.set_title('Expense')
    
    data = list(collection.find({}))

    # Convert 'date' field to datetime
    for entry in data:
        entry['date'] = pd.to_datetime(entry['date'])

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-31')

    # Filter data between start and end dates
    filtered_data = [entry for entry in data if start_date <= entry['date'] <= end_date]

    date = [entry['date'] for entry in filtered_data]
    income = []
    expense = []

    for entry in filtered_data:
        if entry['type'] == 'Income':
            income.append(entry['amount'])
            expense.append(0)  # Replace None with 0 for expense
        else:
            income.append(0)  # Replace None with 0 for income
            expense.append(entry['amount'])
    
    axis2.bar(date, expense, color='red', label='Expense')
    #axis2.fill_between(date, np.nan_to_num(expense), color='lightcoral')
    axis2.set_xlabel('Date')
    axis2.set_ylabel('Expense ($)')
    axis2.legend()
    

    output = BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)

    return send_file(output, mimetype='image/png') 

@app.route('/plot2.png')
def plot2_png():
    fig = Figure(figsize=(10, 6))  # Adjust figsize here if needed

    # Adjust bottom spacing to reduce whitespace
    fig.subplots_adjust(top=0.85, bottom=0.15)  

    # Add subplot for income vs expense comparison
    axis3 = fig.add_subplot(4, 1, 3)  # 3 rows, 1 column, subplot 3
    axis3.set_title('Income vs Expense')

    data = list(collection.find({}))

    # Convert 'date' field to datetime
    for entry in data:
        entry['date'] = pd.to_datetime(entry['date'])

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-31')

    # Filter data between start and end dates
    filtered_data = [entry for entry in data if start_date <= entry['date'] <= end_date]

    date = [entry['date'] for entry in filtered_data]
    income = []
    expense = []

    for entry in filtered_data:
        if entry['type'] == 'Income':
            income.append(entry['amount'])
            expense.append(0)  # Replace None with 0 for expense
        else:
            income.append(0)  # Replace None with 0 for income
            expense.append(entry['amount'])
    # Plot income vs expense comparison
    axis3.bar(date, income, color='green', label='Income')
    axis3.bar(date, expense, color='red', label='Expense')
    
    income_exceeds_expense = np.array(income) > np.array(expense)
    expense_exceeds_income = np.array(expense) > np.array(income)

    # Fill between income and expense curves where income exceeds expense with light green color
    #axis3.fill_between(date, np.nan_to_num(income), np.nan_to_num(expense), where=income_exceeds_expense, color='lightgreen', interpolate=True)

    # Fill between income and expense curves where expense exceeds income with light red color
    #axis3.fill_between(date, np.nan_to_num(income), np.nan_to_num(expense), where=expense_exceeds_income, color='lightcoral', interpolate=True)

    axis3.set_xlabel('Date')
    axis3.set_ylabel('Amount ($)')
    axis3.legend()  

    output = BytesIO()
    fig.savefig(output, format='png', bbox_inches='tight')
    output.seek(0)

    return send_file(output, mimetype='image/png')

# Barchart for categories

@app.route('/plot3.png')
def plot3_png():
    fig = Figure(figsize=(10, 7))  # Adjust figure size if needed
    fig.subplots_adjust(bottom=0.15)
    
    # Add subplot for expense categories comparison
    axis4 = fig.add_subplot(1, 1, 1)  # 3 rows, 1 column, subplot 3
    axis4.set_title('Expense Categories') 

    data = list(collection.find({}))

    # Convert 'date' field to datetime
    for entry in data:
        entry['date'] = pd.to_datetime(entry['date'])

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-31')

    # Filter data between start and end dates
    filtered_data = [entry for entry in data if start_date <= entry['date'] <= end_date]

    date = [entry['date'] for entry in filtered_data]
    income = []
    expense = []

    for entry in filtered_data:
        if entry['type'] == 'Income':
            income.append(entry['amount'])
            expense.append(0)  # Replace None with 0 for expense
        else:
            income.append(0)  # Replace None with 0 for income
            expense.append(entry['amount'])

    

    expense_df = pd.DataFrame(filtered_data)
    expense_df = expense_df[expense_df['type'] == 'Expense']
    category_counts = expense_df['category'].value_counts()
    category_names = category_counts.index
    category_values = category_counts.values
    axis4.bar(category_names, category_values, color='blue')
    axis4.set_ylabel('Frequency')

    output = BytesIO()
    fig.savefig(output, format='png', bbox_inches='tight')
    output.seek(0)

    return send_file(output, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)