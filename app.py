from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
import plotly.express as px
from datetime import datetime

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
        'note': note
    }

    mongo.db.incomeandexpense.insert_one(entry)
    return redirect(url_for('index'))

@app.route('/predict_expense')
def predict_expense():
    # Implement prediction logic based on historical data
    # This could involve training a machine learning model on past entries
    # For simplicity, we'll assume a basic rule-based prediction for now

    predicted_expense = 0  # Replace with your prediction logic
    return render_template('predict_expense.html', predicted_expense=predicted_expense)

@app.route('/dashboard')
def dashboard():
    entries = list(mongo.db.entries.find())
    df = px.data.frame_from_dict(entries)

    fig = px.bar(df, x='date', y='amount', color='category', title='Income/Expense Overview')

    graphJSON = px.to_json(fig)
    return render_template('dashboard.html', graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
