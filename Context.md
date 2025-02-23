# Credit Risk AI Workflow

## 1. Project Structure:

CreditRiskAI/
├── data/                  
│   ├── train.csv
│   └── test.csv
├── app.py
├── models.py
├── frontend/              # React frontend
│   ├── public/            # Static assets
│   │   └── index.html     # HTML template
│   ├── src/               # React source code
│   │   ├── components/    # Reusable components
│   │   ├── App.js         # Main application component
│   │   ├── App.css        # Styles for App.js
│   │   ├── index.js       # Entry point for React
│   │   └── api.js         # API utility for backend communication
└── 

## 3. Database Schema
**Training Data Metadata**:
- ID (Unique record identifier)
- Customer_ID (Client tracking code)
- Month (Reporting period)
- Name (Customer full name)
- Age (Current age)
- SSN (Masked social security number)
- Occupation (Employment category)
- Annual_Income (Yearly earnings)
- Monthly_Inhand_Salary (Net monthly income)
- Num_Bank_Accounts (Active accounts count)
- Num_Credit_Card (Active credit cards)
- Interest_Rate (Current APR)
- Num_of_Loan (Active loans)
- Type_of_Loan (Mortgage/Personal/Auto)
- Delay_from_due_date (Avg payment delay days)
- Num_of_Delayed_Payment (Late payments count)
- Changed_Credit_Limit (Recent limit changes)
- Num_Credit_Inquiries (Credit checks)
- Credit_Mix (Credit portfolio diversity)
- Outstanding_Debt (Current liabilities)
- Credit_Utilization_Ratio (Used credit %)
- Credit_History_Age (Months since first credit)
- Payment_of_Min_Amount (Minimum payment compliance)
- Total_EMI_per_month (Monthly installments)
- Amount_invested_monthly (Savings contributions)
- Payment_Behaviour (Spending patterns)
- Monthly_Balance (Liquid assets)
- Credit_Score (Target variable: Poor/Standard/Good)

## 5. Production Deployment
- **Database**: PostgreSQL
- **CI/CD**: GitHub Actions → Cloud Build → Cloud Run
- **Monitoring**:
  - Prometheus metrics endpoint
  - Cloud Logging integration

## Workflow Recap
1. Data Collection → 2. Feature Store → 3. Model Training →  
4. API Deployment → 5. Performance Monitoring → 6. Retraining
