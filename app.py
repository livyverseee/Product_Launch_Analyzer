from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
import os
import json

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# Ensure static folder exists using absolute path (needed on Render)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
CHART_PATH = os.path.join(STATIC_DIR, 'profit_chart.png')

# ==========================
# 🔐 CONFIGURE GROQ API
# ==========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. Add it to your .env file.")
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================
# 📊 DIGITAL TWIN SIMULATION
# ==========================
def simulate_digital_twin(price, demand, competition, budget, scenario):
    # Normalize inputs to a base unit scale (independent of raw values)
    base_units = 100  # start with 100 base units sold per month
    cost_per_unit = price * 0.38
    monthly_marketing_cost = budget / 12

    # Scenario multipliers
    if scenario == "aggressive_marketing":
        budget *= 1.5
        monthly_marketing_cost = budget / 12
        demand_boost = 1.2
    elif scenario == "price_drop":
        price *= 0.85
        cost_per_unit = price * 0.38
        demand_boost = 1.3
    else:
        demand_boost = 1.0

    # Demand attractiveness (0.3 to 1.0)
    demand_factor = 0.3 + (demand / 100) * 0.7

    # Competition resistance (high competition = fewer units)
    competition_resistance = 1 - (competition / 100) * 0.6

    # Marketing power (how much budget boosts units)
    marketing_power = min(2.0, 1 + (budget / 500000))

    simulation = []

    for month in range(1, 13):
        # S-curve growth: slow start, accelerates mid, plateaus late
        # Uses logistic-style growth
        growth_multiplier = 1 + (month / 12) * 1.8  # grows from 1x to 2.8x

        # Word of mouth accelerates from month 3
        wom = 1.0 + max(0, (month - 2) * 0.08)
        wom = min(wom, 2.2)

        # Seasonality
        seasonality = 1 + 0.1 * np.sin(month * np.pi / 6)

        # Units sold — always grows with time
        units_sold = (
            base_units
            * demand_factor
            * demand_boost
            * competition_resistance
            * marketing_power
            * growth_multiplier
            * wom
            * seasonality
        )

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
# 📈 GENERATE PROFIT CHART
# ==========================
def generate_profit_chart(simulation_data):
    import io, base64

    months = [d["month"] for d in simulation_data]
    profits = [d["profit"] for d in simulation_data]
    revenues = [d["revenue"] for d in simulation_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.tick_params(colors='#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')

    profit_colors = ['#3fb950' if p > 0 else '#f85149' for p in profits]
    ax1.bar(months, profits, color=profit_colors, alpha=0.85, width=0.7)
    ax1.axhline(y=0, color='#484f58', linestyle='--', linewidth=1)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Profit (₹)")
    ax1.set_title("12-Month Profit Forecast")
    ax1.set_xticks(months)

    ax2.fill_between(months, revenues, alpha=0.2, color='#58a6ff')
    ax2.plot(months, revenues, color='#58a6ff', linewidth=2.5, marker='o', markersize=5)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Revenue (₹)")
    ax2.set_title("12-Month Revenue Trend")
    ax2.set_xticks(months)

    plt.tight_layout(pad=3)

    # Save to memory buffer instead of file — no caching issues!
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return chart_b64


# ==========================
# 🤖 GROQ AI STRATEGY
# ==========================
def generate_ai_strategy(product, price, demand, competition, budget, score, risk, simulation):
    try:
        peak_profit = max(d["profit"] for d in simulation)
        avg_profit = sum(d["profit"] for d in simulation) / len(simulation)
        profitable_months = sum(1 for d in simulation if d["profit"] > 0)

        prompt = f"""You are an elite startup advisor and go-to-market strategist based in India.

Analyze this product launch data and provide actionable strategic advice in Indian market context:

Product: {product}
Pricing: ₹{price}
Market Demand Score: {demand}/100
Competition Intensity: {competition}/100
Monthly Marketing Budget: ₹{budget}
Success Score: {score}/100
Risk Classification: {risk}

Digital Twin Simulation Results (12-month forecast):
- Peak Monthly Profit: ₹{peak_profit:,.0f}
- Average Monthly Profit: ₹{avg_profit:,.0f}
- Profitable Months: {profitable_months}/12

Respond ONLY with this JSON object, no markdown, no extra text:
{{
  "summary": "2-sentence executive summary of the opportunity",
  "top_risks": ["risk 1", "risk 2", "risk 3"],
  "quick_wins": ["action 1 (do this week)", "action 2 (do this week)", "action 3 (do this week)"],
  "90_day_plan": ["Month 1: ...", "Month 2: ...", "Month 3: ..."],
  "pricing_advice": "specific pricing strategy recommendation in Indian Rupees",
  "marketing_advice": "specific Indian marketing channel and budget allocation advice",
  "competitive_moat": "how to build defensibility against competition in Indian market"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a startup business strategist. Always respond with valid JSON only, no extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        print("Groq API ERROR:", e)
        return {
            "summary": f"API Error: {str(e)[:120]}",
            "top_risks": ["Check GROQ_API_KEY in .env", "Ensure groq package is installed", "Verify API key at console.groq.com"],
            "quick_wins": ["pip install groq", "Add GROQ_API_KEY=gsk_... to .env", "Restart Flask server"],
            "90_day_plan": ["Month 1: Fix API connection", "Month 2: Rerun analysis", "Month 3: Execute strategy"],
            "pricing_advice": "Fix API key to get pricing recommendations.",
            "marketing_advice": "Fix API key to get marketing recommendations.",
            "competitive_moat": "Fix API key to get competitive analysis."
        }


# ==========================
# 🏠 ROUTES
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    product = request.form["product"]
    price = float(request.form["price"])
    demand = max(20.0, float(request.form.get("demand", 60)))
    competition = max(20.0, float(request.form.get("competition", 50)))
    budget = float(request.form["budget"])
    print(f"DEBUG: price={price}, demand={demand}, competition={competition}, budget={budget}")
    scenario = request.form["scenario"]

    price_score = min(100, max(0, 100 - (price / 1000)))
    demand_score = demand
    competition_score = 100 - competition
    budget_score = min(100, budget / 50000)

    score = round(
        demand_score * 0.35 +
        competition_score * 0.30 +
        budget_score * 0.20 +
        price_score * 0.15,
        1
    )
    score = max(0, min(100, score))

    if score >= 72:
        risk = "Low Risk"
        market_fit = "Excellent"
        risk_color = "green"
    elif score >= 50:
        risk = "Medium Risk"
        market_fit = "Good"
        risk_color = "yellow"
    else:
        risk = "High Risk"
        market_fit = "Needs Work"
        risk_color = "red"

    simulation = simulate_digital_twin(price, demand, competition, budget, scenario)
    chart_b64 = generate_profit_chart(simulation)

    peak_profit = max(d["profit"] for d in simulation)
    total_revenue = sum(d["revenue"] for d in simulation)
    profitable_months = sum(1 for d in simulation if d["profit"] > 0)
    final_profit = simulation[-1]["profit"]

    strategy = generate_ai_strategy(product, price, demand, competition, budget, score, risk, simulation)

    return render_template(
        "result.html",
        product=product,
        score=score,
        risk=risk,
        risk_color=risk_color,
        market_fit=market_fit,
        peak_profit=f"{peak_profit:,.0f}",
        total_revenue=f"{total_revenue:,.0f}",
        profitable_months=profitable_months,
        final_profit=f"{final_profit:,.0f}",
        strategy=strategy,
        simulation=simulation[:6],
        scenario=scenario.replace("_", " ").title(),
        chart_b64=chart_b64
    )


if __name__ == "__main__":
    app.run(debug=True)