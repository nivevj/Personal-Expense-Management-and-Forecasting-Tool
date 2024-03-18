from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_pymongo import PyMongo
from datetime import datetime
from pymongo import MongoClient
from prophet import Prophet
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure

app = Flask(__name__)

app.config['MONGO_URI'] = '<your mongo uri>'
mongo = PyMongo(app)

mongo_uri = "<your mongo uri>"
client = MongoClient(mongo_uri)
db = client.expense_tracking
collection = db.transactions

@app.route('/')
def index():
    return render_template('home.html')

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
    
    predicted_expense = next_month_prediction
    return render_template('predict_expense.html', predicted_expense=predicted_expense,avg_spending_category=avg_spending_category) 

@app.route('/transactions_this_month')
def transactions_this_month():
    transactions = list(mongo.db.transactions.find({
        '$or': [{'type': 'Expense'}, {'type': 'Income'}]
    }))
    return render_template('transactions_month.html', transactions=transactions)

@app.route('/plot.png')
def plot_png():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    data = list(collection.find({}))
    # Convert 'date' field to datetime
    for entry in data:
        entry['date'] = pd.to_datetime(entry['date'])
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-31')
    # Filter data between start and end dates
    filtered_data = [entry for entry in data if start_date <= entry['date'] <= end_date]
    date = [entry['date'] for entry in filtered_data]
    expense = [entry['amount'] for entry in filtered_data]
    axis.plot(date, expense, color='blue', label='Expense')
    num_ticks = 10  # Adjust the number of ticks as needed
    num_points = len(date)
    tick_spacing = num_points // num_ticks
    x_ticks = date[::tick_spacing]  # Select ticks with spacing
    x_tick_labels = [tick.strftime('%Y-%m-%d') for tick in x_ticks]  # Format tick labels
    
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labels, rotation=90)  #
    axis.set_xlabel('Date')
    axis.set_ylabel('Amount ($)')
    axis.legend()
    output = BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)