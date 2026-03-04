from flask import Flask, render_template, request, redirect, url_for, flash, session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
import os, json, io, base64
from datetime import datetime
from functools import wraps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Flask-Login & DB
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "producttwin-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access ProductTwin."

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================
# 👤 USER MODEL
# ==========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# ==========================
# 📊 DIGITAL TWIN SIMULATION
# ==========================
def simulate_digital_twin(price, demand, competition, budget, scenario):
    base_units = 100
    cost_per_unit = price * 0.38
    monthly_marketing_cost = budget / 12
    demand_boost = 1.0

    if scenario == "aggressive_marketing":
        budget *= 1.5
        monthly_marketing_cost = budget / 12
        demand_boost = 1.2
    elif scenario == "price_drop":
        price *= 0.85
        cost_per_unit = price * 0.38
        demand_boost = 1.3
    elif scenario == "market_expansion":
        demand_boost = 1.15
        monthly_marketing_cost = budget / 12 * 0.9
    elif scenario == "lean_launch":
        monthly_marketing_cost = budget / 12 * 0.5
        demand_boost = 0.85
    elif scenario == "viral_growth":
        demand_boost = 1.4
        monthly_marketing_cost = budget / 12 * 1.1

    demand_factor = 0.3 + (max(20.0, demand) / 100) * 0.7
    competition_resistance = 1 - (max(20.0, competition) / 100) * 0.6
    marketing_power = min(2.0, 1 + (budget / 500000))

    simulation = []
    for month in range(1, 13):
        growth_multiplier = 1 + (month / 12) * 1.8
        wom = min(1.0 + max(0, (month - 2) * 0.08), 2.2)
        seasonality = 1 + 0.1 * np.sin(month * np.pi / 6)
        units_sold = (base_units * demand_factor * demand_boost *
                      competition_resistance * marketing_power *
                      growth_multiplier * wom * seasonality)
        revenue = units_sold * price
        production_cost = units_sold * cost_per_unit
        profit = revenue - production_cost - monthly_marketing_cost
        simulation.append({
            "month": month,
            "profit": round(profit, 2),
            "revenue": round(revenue, 2),
            "units_sold": round(units_sold, 1),
        })
    return simulation

# ==========================
# 📈 GENERATE CHART (base64)
# ==========================
def generate_profit_chart(simulation_data):
    months = [d["month"] for d in simulation_data]
    profits = [d["profit"] for d in simulation_data]
    revenues = [d["revenue"] for d in simulation_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0f1e')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#111827')
        for spine in ax.spines.values():
            spine.set_color('#1f2937')
        ax.tick_params(colors='#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        ax.xaxis.label.set_color('#9ca3af')
        ax.title.set_color('#f1f5f9')

    profit_colors = ['#10b981' if p > 0 else '#ef4444' for p in profits]
    bars = ax1.bar(months, profits, color=profit_colors, alpha=0.9, width=0.7)
    ax1.axhline(y=0, color='#374151', linestyle='--', linewidth=1)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Profit (₹)")
    ax1.set_title("12-Month Profit Forecast", fontweight='bold', pad=15)
    ax1.set_xticks(months)

    ax2.fill_between(months, revenues, alpha=0.15, color='#6366f1')
    ax2.plot(months, revenues, color='#6366f1', linewidth=2.5, marker='o', markersize=6)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Revenue (₹)")
    ax2.set_title("12-Month Revenue Trend", fontweight='bold', pad=15)
    ax2.set_xticks(months)

    plt.tight_layout(pad=3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ==========================
# 🤖 GROQ AI STRATEGY
# ==========================
def generate_ai_strategy(product, price, demand, competition, budget, score, risk, simulation):
    try:
        peak_profit = max(d["profit"] for d in simulation)
        avg_profit = sum(d["profit"] for d in simulation) / len(simulation)
        profitable_months = sum(1 for d in simulation if d["profit"] > 0)

        prompt = f"""You are an elite startup advisor for the Indian market.

Product: {product}
Price: ₹{price} | Demand Score: {demand}/100 | Competition: {competition}/100
Marketing Budget: ₹{budget}/month | Success Score: {score}/100 | Risk: {risk}

Digital Twin Simulation (12-month):
- Peak Monthly Profit: ₹{peak_profit:,.0f}
- Avg Monthly Profit: ₹{avg_profit:,.0f}
- Profitable Months: {profitable_months}/12

Respond ONLY with this JSON (no markdown):
{{
  "summary": "2-sentence executive summary",
  "ai_insight": "1 surprising insight the founder might not have considered",
  "top_risks": ["risk 1", "risk 2", "risk 3"],
  "quick_wins": ["action 1", "action 2", "action 3"],
  "90_day_plan": ["Month 1: ...", "Month 2: ...", "Month 3: ..."],
  "pricing_advice": "specific ₹ pricing strategy",
  "marketing_advice": "specific Indian channel + budget split advice",
  "competitive_moat": "how to build defensibility in Indian market",
  "digital_twin_interpretation": "what the simulation numbers mean for this specific product"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a startup strategist. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("Groq ERROR:", e)
        return {
            "summary": f"Error: {str(e)[:100]}",
            "ai_insight": "Fix API key to get insights.",
            "top_risks": ["API issue"], "quick_wins": ["Fix API"],
            "90_day_plan": ["Fix API"], "pricing_advice": "N/A",
            "marketing_advice": "N/A", "competitive_moat": "N/A",
            "digital_twin_interpretation": "N/A"
        }

# ==========================
# 🔐 AUTH ROUTES
# ==========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            return redirect(url_for("home"))
        flash("Invalid email or password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not name or not email or not password:
            flash("All fields are required.")
        elif password != confirm:
            flash("Passwords do not match.")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.")
        else:
            user = User(name=name, email=email, password=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("home"))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ==========================
# 🏠 MAIN ROUTES
# ==========================
@app.route("/")
@login_required
def home():
    return render_template("index.html", user=current_user)

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    product = request.form["product"]
    price = float(request.form["price"])
    demand = max(20.0, float(request.form.get("demand", 60)))
    competition = max(20.0, float(request.form.get("competition", 50)))
    budget = float(request.form["budget"])
    scenario = request.form["scenario"]

    price_score = min(100, max(0, 100 - (price / 1000)))
    demand_score = demand
    competition_score = 100 - competition
    budget_score = min(100, budget / 50000)

    score = round(demand_score*0.35 + competition_score*0.30 + budget_score*0.20 + price_score*0.15, 1)
    score = max(0, min(100, score))

    if score >= 72:
        risk, market_fit, risk_color = "Low Risk", "Excellent", "green"
    elif score >= 50:
        risk, market_fit, risk_color = "Medium Risk", "Good", "yellow"
    else:
        risk, market_fit, risk_color = "High Risk", "Needs Work", "red"

    simulation = simulate_digital_twin(price, demand, competition, budget, scenario)
    chart_b64 = generate_profit_chart(simulation)

    peak_profit = max(d["profit"] for d in simulation)
    total_revenue = sum(d["revenue"] for d in simulation)
    profitable_months = sum(1 for d in simulation if d["profit"] > 0)
    final_profit = simulation[-1]["profit"]

    strategy = generate_ai_strategy(product, price, demand, competition, budget, score, risk, simulation)

    return render_template("result.html",
        user=current_user,
        product=product, score=score, risk=risk,
        risk_color=risk_color, market_fit=market_fit,
        peak_profit=f"{peak_profit:,.0f}",
        total_revenue=f"{total_revenue:,.0f}",
        profitable_months=profitable_months,
        final_profit=f"{final_profit:,.0f}",
        strategy=strategy, simulation=simulation[:6],
        scenario=scenario.replace("_", " ").title(),
        chart_b64=chart_b64,
        price=price, demand=demand, competition=competition, budget=budget
    )

if __name__ == "__main__":
    app.run(debug=True)