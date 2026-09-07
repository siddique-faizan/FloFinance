# FloFinance — Personal Finance Tracker with ML Prediction

A full-stack web application for tracking personal expenses with machine learning powered spending predictions.

## Features

- User authentication - register, login, logout with hashed passwords
- Add expenses by category - Food, Transport, Housing, Entertainment, Health, Shopping, Education
- Interactive charts - category doughnut, bar chart breakdown
- ML prediction - Linear Regression model trained on spending history predicts next month's expenses
- Actual vs predicted spending graph - visualises model accuracy over time
- Persistent SQLite database via SQLAlchemy ORM

## Tech Stack

- **Backend** - Flask, SQLAlchemy, Flask-Login
- **ML** - Scikit-learn (Linear Regression), NumPy
- **Frontend** - HTML, CSS, JavaScript, Chart.js
- **Database** - SQLite
- **Deployment** - Railway

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`

## How the ML prediction works

Monthly spending totals are extracted from the database and used as training data for a Scikit-learn Linear Regression model. The model learns the trend in spending over time and predicts the next month's total. A minimum of 3 months of data is required before predictions activate.

## Deployment

Deployed live on Railway. Connected via GitHub for automatic deployments on push.

Independent project - built as part of a full-stack and ML learning journey.
