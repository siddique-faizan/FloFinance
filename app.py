from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# ── Models ────────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    expenses = db.relationship('Expense', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200))
    date = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── ML Prediction ─────────────────────────────────────────────────────────────

def predict_next_month(user_id):
    expenses = Expense.query.filter_by(user_id=user_id).all()

    if len(expenses) < 3:
        return None, None, None

    monthly = defaultdict(float)
    for expense in expenses:
        key = (expense.date.year, expense.date.month)
        monthly[key] += expense.amount

    sorted_months = sorted(monthly.keys())

    if len(sorted_months) < 2:
        return None, None, None

    X = np.array(range(len(sorted_months))).reshape(-1, 1)
    y = np.array([monthly[m] for m in sorted_months])

    model = LinearRegression()
    model.fit(X, y)

    next_month_index = np.array([[len(sorted_months)]])
    prediction = model.predict(next_month_index)[0]

    labels = [f"{m[0]}/{m[1]:02d}" for m in sorted_months]
    actuals = y.tolist()

    return round(prediction, 2), labels, actuals


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid email or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date.desc()).limit(10).all()
    total = db.session.query(db.func.sum(Expense.amount)).filter_by(user_id=current_user.id).scalar() or 0

    category_totals = db.session.query(
        Expense.category,
        db.func.sum(Expense.amount)
    ).filter_by(user_id=current_user.id).group_by(Expense.category).all()

    prediction, labels, actuals = predict_next_month(current_user.id)

    return render_template('dashboard.html',
        expenses=expenses,
        total=round(total, 2),
        category_totals=category_totals,
        prediction=prediction,
        labels=labels,
        actuals=actuals
    )


@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_expense():
    categories = ['Food', 'Transport', 'Housing', 'Entertainment', 'Health', 'Shopping', 'Education', 'Other']

    if request.method == 'POST':
        amount = float(request.form.get('amount'))
        category = request.form.get('category')
        description = request.form.get('description')
        date_str = request.form.get('date')
        date = datetime.strptime(date_str, '%Y-%m-%d') if date_str else datetime.utcnow()

        expense = Expense(
            user_id=current_user.id,
            amount=amount,
            category=category,
            description=description,
            date=date
        )
        db.session.add(expense)
        db.session.commit()
        flash('Expense added.', 'success')
        return redirect(url_for('dashboard'))

    return render_template('add.html', categories=categories)


@app.route('/delete/<int:expense_id>')
@login_required
def delete_expense(expense_id):
    expense = Expense.query.get_or_404(expense_id)
    if expense.user_id != current_user.id:
        flash('Unauthorised.', 'error')
        return redirect(url_for('dashboard'))
    db.session.delete(expense)
    db.session.commit()
    flash('Expense deleted.', 'success')
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=8080)
