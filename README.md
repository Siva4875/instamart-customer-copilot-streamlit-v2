# Instamart Customer Intelligence & Support Copilot

## Streamlit Portfolio Edition – Version 2.0

An interactive customer intelligence and support analytics application built using Python, Streamlit, Pandas, Plotly and Scikit-learn.

## Project Overview

This application provides five interactive modules:

1. Executive Dashboard
2. Customer Intelligence
3. Churn Prediction & Retention Intelligence
4. Support Ticket Management
5. AI Support Copilot

## Application Features

### 1. Executive Dashboard
- Customer and order KPIs
- Completed order value
- Order status analysis
- Customer segment distribution
- Monthly order trends
- Support ticket and refund-agent evaluation summaries

### 2. Customer Intelligence
- RFM-based customer segmentation
- Customer segment distribution
- Recency versus monetary analysis
- Customer demographics by city and gender
- Interactive customer filters
- Customer details and CSV export

### 3. Churn Prediction
- Saved Scikit-learn lapse prediction model
- Historical held-out evaluation
- Customer lapse probabilities
- Low, Medium and High descriptive risk categories
- Actual versus predicted outcomes
- Customer-level filtering and CSV export

Note: Predictions represent historical held-out evaluation results, not live customer churn predictions. Risk categories are descriptive presentation bands.

### 4. Support Ticket Management
- Ticket processing status analysis
- Classified, flagged and skipped ticket counts
- Ticket category distribution
- Priority and routing analysis
- Ticket search and filters
- Downloadable CSV reports

### 5. AI Support Copilot
- Searches five policy documents
- TF-IDF-based policy retrieval
- Cosine similarity ranking
- Relevant policy excerpts
- Source filenames and similarity scores
- Relevance filtering and unsupported-question fallback

Note: This is an offline policy retrieval demonstration, not a hosted generative AI service. Similarity scores are not answer-confidence scores. Human verification is required for sensitive support decisions.

The public dashboard does not execute refunds or modify customer records.

## Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Dashboard | Streamlit |
| Data processing | Pandas, NumPy |
| Visualizations | Plotly |
| Machine Learning | Scikit-learn |
| Model persistence | Joblib |
| Policy retrieval | TF-IDF, cosine similarity |
| Data storage | CSV, SQLite |

## Repository Structure

```text
INSTAMART_STREAMLIT_V2/
├── app.py
├── requirements.txt
├── README.md
├── generate_data.py
├── run_project.py
├── mock_llm.py
├── data/
│   ├── clean/
│   ├── raw/
│   └── policies/
├── outputs/
├── src/
├── notebooks/
├── sql/
└── state/
```

## Installation and Local Setup

### Prerequisites
- Python 3.12 recommended
- Git

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd INSTAMART_STREAMLIT_V2
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the application

```powershell
python -m streamlit run app.py
```

Open:

`http://localhost:8501`

## Streamlit Cloud Deployment

1. Publish the V2 project to a separate GitHub repository.
2. Connect your GitHub account to Streamlit Community Cloud.
3. Select the V2 repository.
4. Choose the `main` branch.
5. Set the application entry point to `app.py`.
6. Deploy the application.

## Limitations

- Uses synthetic project data.
- No live production Instamart database connection.
- No live customer transactions.
- No hosted generative AI integration.
- No actual refund execution through the public dashboard.
- Historical model evaluation does not guarantee future performance.

## Portfolio Highlights

This project demonstrates:

- Python application development
- Data analysis and visualization
- Customer segmentation
- Machine-learning model integration
- Support ticket analytics
- Retrieval-based document search
- Interactive Streamlit dashboards
- Git and GitHub project management
- Deployment preparation

## Project Status

**Version:** 2.0 – Streamlit Portfolio Edition

**Dashboard pages:** 5 implemented and visually reviewed.

**Next milestone:** Separate GitHub publication and hosted deployment.