# Telco Customer Churn Predictor

> A recruiter-friendly, deployable machine learning project that turns raw telco customer data into a live churn-risk application with interactive what-if analysis and batch scoring.

![App Screenshot](assets/app-screenshot.png)

**Live Demo:** https://telco-churn-predictor.vercel.app

## Why This Project Matters

Customer churn is one of the clearest business problems in subscription-based industries.
If a telco provider can identify which customers are most likely to leave, it can intervene early with pricing, support, bundling, or retention offers before revenue is lost.

This project takes that business problem and turns it into something concrete:

- a cleaned and reproducible machine learning workflow
- a deployed Flask application for real-time scoring
- a batch upload tool for business users
- an interactive simulator to explore how churn drivers change the prediction

This is not just a notebook with charts. It is a full project that connects:

1. business problem framing
2. data preparation
3. model evaluation
4. application design
5. deployment readiness

## Quick Snapshot

- **Project theme:** Telco churn prediction
- **Primary goal:** Predict which customers are at risk of leaving
- **Best deployed model:** XGBoost
- **Validation ROC AUC:** 85.14%
- **Validation Accuracy:** 80.20%
- **Batch scoring formats:** CSV, XLSX, JSON, and structured-table PDF
- **Interactive features:** Single-customer what-if simulator with adjustable inputs
- **Live hosting status:** Deployed on Vercel

## What Makes This Repo Strong From a Recruiter Perspective

This repository is designed to show more than model training.
It demonstrates the kind of end-to-end thinking that is valuable in data, ML, analytics, and product-oriented engineering roles:

- translating a business problem into a measurable ML objective
- identifying meaningful churn drivers from customer behavior and service choices
- building a user-facing application instead of stopping at model metrics
- documenting tradeoffs honestly instead of inflating results
- preparing the project for deployment so it can be evaluated like a real product

## The Business Problem

Telco churn directly impacts recurring revenue, customer lifetime value, and acquisition cost efficiency.
Acquiring a new customer is usually more expensive than retaining an existing one, so even modest improvements in churn detection can create meaningful business value.

This project focuses on a simple question:

**Can we identify customers with elevated churn risk early enough to support retention decisions?**

Examples of practical actions this model could support:

- targeting customers for retention offers
- prioritizing outreach from account or support teams
- identifying segments with structurally higher churn risk
- testing how contract structure or service bundles may affect churn probability

## What The App Actually Does

The application supports two user journeys.

### 1. Interactive what-if simulator

The home page lets a user adjust customer attributes and immediately see how the predicted churn probability changes.

This is useful for:

- recruiters who want to see that the model is truly wired into an interface
- stakeholders who want a simple demo without reading code
- analysts who want to understand how churn drivers influence the score

Adjustable inputs include:

- gender
- senior citizen status
- partner / dependents
- phone service and multiple lines
- internet service type
- online security / backup / device protection / tech support
- streaming services
- contract type
- paperless billing
- payment method
- tenure months
- monthly charges
- total charges

Important note for authenticity:
the dataset does **not** contain an `age` column.
The closest available age-related field is `Senior Citizen`, so the app uses that instead of pretending age exists in the source data.

### 2. Batch customer scoring

The app also supports scoring a file of customer records at once.

Supported uploads:

- `csv`
- `xlsx`
- `json`
- `pdf` containing a structured table close to the expected schema

For each customer record, the app returns:

- `customerid`
- `churn_probability`
- `predicted_churn`

This turns the project from a simple model demo into something much closer to a business workflow.

## Example Behaviors Observed In Local Testing

I tested the live local app with several contrasting profiles after deploying the cleaned XGBoost pipeline locally.

### Higher-risk example

- month-to-month contract
- fiber optic internet
- low tenure
- higher monthly charges
- limited support/security services

Observed churn probability:

- **64.24%** churn probability
- predicted churn: **Yes**

### Very high-risk example

- senior citizen
- month-to-month contract
- fiber optic service
- very short tenure
- high monthly charges
- no online security or tech support

Observed churn probability:

- **91.26%** churn probability
- predicted churn: **Yes**

### Lower-risk example

- long tenure
- two-year contract
- lower monthly charges
- stronger support / protection services
- automatic payment setup

Observed churn probability:

- **0.20%** churn probability
- predicted churn: **No**

These examples make the demo more intuitive and show that the application responds meaningfully to parameter changes.

## Model Development Approach

The original repository included notebook-driven experimentation.
This version was reworked into a cleaner and more reliable deployment pipeline so the app and the documentation reflect the same truth.

### Models compared

Three models are trained and evaluated in the current workflow:

- Logistic Regression
- Random Forest
- XGBoost

### Shared preprocessing pipeline

Each candidate model uses structured preprocessing before training:

- numeric imputation with median values
- categorical imputation with most-frequent values
- one-hot encoding for categorical variables
- train/test split before evaluation

This matters because earlier notebook-style workflows can accidentally inflate performance if resampling or transformation is applied before the train/test split.

## Final Model Choice

The currently selected deployed model is **XGBoost**.

It was chosen because, in the cleaned evaluation flow used by the application, it produced the strongest ROC AUC while also leading in overall accuracy.

### Current local validation results

#### XGBoost

- ROC AUC: **85.14%**
- Average Precision: **66.70%**
- Accuracy: **80.20%**
- Precision: **64.62%**
- Recall: **56.15%**
- F1: **60.09%**

#### Random Forest

- ROC AUC: **84.75%**
- Accuracy: **78.99%**
- F1: **62.53%**

#### Logistic Regression

- ROC AUC: **84.89%**
- Accuracy: **74.31%**
- F1: **61.73%**

### Why not choose only the highest F1 model?

Because model choice depends on the business objective.
For this project, the active selection rule favors:

1. higher ROC AUC
2. then F1 as a tiebreaker

That makes XGBoost the most appropriate deployed default in the current setup.
If the business objective changed to prioritize recall or a different thresholding strategy, the selection rule could also change.

## Why The Results Are More Credible Now

One of the most important improvements in this repo is that the evaluation has been made more honest.

This version avoids several problems that often make portfolio ML projects look weaker to experienced reviewers:

- inflated metrics caused by train/test leakage
- mismatch between notebook results and deployed app logic
- unrealistic claims that are not supported by the actual code
- vague “AI-powered” language without concrete implementation details

Instead, this repo now aims to be credible:

- the README matches the current codebase
- the deployed model is the model actually used by the app
- the UI inputs reflect real fields from the dataset
- limitations are stated clearly

## Features Used By The Model

The app currently uses the following predictive inputs:

- `gender`
- `senior_citizen`
- `partner`
- `dependents`
- `phone_service`
- `multiple_lines`
- `internet_service`
- `online_security`
- `online_backup`
- `device_protection`
- `tech_support`
- `streaming_tv`
- `streaming_movies`
- `contract`
- `paperless_billing`
- `payment_method`
- `tenure_months`
- `monthly_charges`
- `total_charges`

## Files Accepted By The App

The minimal model-ready schema is based on those features above.

The app also accepts the original dataset-style headers, such as:

- `CustomerID`
- `Senior Citizen`
- `Monthly Charges`
- `Total Charges`
- `Churn Label`

This makes the app easier to demo with the original dataset and also easier to adapt to cleaned downstream files.

## Repository Structure

```text
Teleco-Customer-Churn-Prediction/
├── app.py                     # Flask app routes and page rendering
├── modeling.py                # Training, preprocessing, prediction helpers
├── train_model.py             # Rebuilds the saved model artifact
├── wsgi.py                    # Production entrypoint
├── requirements.txt           # Python dependencies
├── artifacts/
│   ├── churn_model.joblib     # Saved trained model bundle
│   └── metrics.json           # Validation metrics for all candidate models
├── templates/
│   └── index.html             # Main app UI
├── static/
│   └── styles.css             # App styling
├── assets/
│   └── app-screenshot.png     # Real screenshot from the app
└── Telco_customer_churn.xlsx  # Source dataset
```

## Running The Project Locally

```bash
pip install -r requirements-dev.txt
python train_model.py
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

Health endpoint:

```text
http://localhost:5000/health
```

## Deployment

### Vercel

Live app:

- https://telco-churn-predictor.vercel.app

This deployment works because the runtime was refactored to use a lightweight ONNX inference artifact instead of shipping the full training-time XGBoost stack into the serverless environment.
That keeps the deployed model behavior aligned with XGBoost while making the bundle small enough for Vercel.

### Render

The repository also includes a `render.yaml` configuration for deployment on Render.
Render remains a valid alternative deployment target for teams that prefer a more traditional Python hosting flow.

## What A Senior Recruiter Should Notice

This project shows a few qualities that are often missing in early portfolio repositories:

- it is easy to understand what the project is solving
- the README is written for both technical and non-technical readers
- the code supports the story the README is telling
- the model is not treated like a black box without context
- the app creates a realistic demonstration path for reviewers

## Honest Limitations

To keep the repository authentic, here are the current limitations plainly stated:

- the dataset is a structured churn dataset, not a live production feed
- PDF parsing is best-effort and works only when the file contains a usable table
- the app does not yet provide row-level model explanations such as SHAP values
- the deployed threshold is currently the default probability threshold rather than a business-optimized custom threshold
- long-term production concerns such as authentication, monitoring, and audit logging are outside this demo scope

## High-Value Next Steps

If this project were taken one step further, the best improvements would be:

1. add feature-level explanation for each prediction
2. expose threshold tuning based on business cost of false positives vs false negatives
3. add downloadable sample templates for upload testing
4. store scored files and prediction history for authenticated users
5. add automated tests for preprocessing, uploads, and prediction routes

## Recruiter-Focused Summary

If you only read one section, this is the short version:

This repository is an end-to-end churn prediction project that moves beyond notebook experimentation into a working application.
It combines business framing, model evaluation, UI design, file-based scoring, and deployment readiness in one coherent story.
The strongest signal in this repo is not just that a model was trained, but that the project was shaped into something a real stakeholder could understand, try, and discuss.
