# 🧠 Mental Health in Tech | Advanced Exploratory Data Analysis & Executive Analytics Suite

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mental-health-in-tech-survey-pe2zagjv2otwdymbfqmqpc.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-ML_Modeling-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ankit-604/Mental-Health-in-Tech-Survey)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

> ### 🌐 **Live Cloud Deployment**
> 🔗 **Interactive Dashboard:** **[mental-health-tech-analytics.streamlit.app](https://mental-health-in-tech-survey-pe2zagjv2otwdymbfqmqpc.streamlit.app/)**  
> 🔗 **GitHub Repository:** **[github.com/Number789Alpha/Mental-Health-in-Tech-Survey](https://github.com/Ankit-604/Mental-Health-in-Tech-Survey)**

---

## 📑 Table of Contents
- [🌐 Live Deployment](#-live-cloud-deployment)
- [📌 Executive Summary](#-executive-summary)
- [🎯 Problem Statement & Business Objectives](#-problem-statement--business-objectives)
- [📊 Dataset Architecture & Preprocessing Pipeline](#-dataset-architecture--preprocessing-pipeline)
- [🌟 Key Discoveries & Empirical Insights](#-key-discoveries--empirical-insights)
- [🔬 Statistical Hypothesis Testing & Imbalance Auditing](#-statistical-hypothesis-testing--imbalance-auditing)
- [🤖 Machine Learning & AI Risk Simulator](#-machine-learning--ai-risk-simulator)
- [🖥️ Streamlit Executive Dashboard Guide](#️-streamlit-executive-dashboard-guide)
- [💡 Strategic ROI Blueprint & Enterprise Pillars](#-strategic-roi-blueprint--enterprise-pillars)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Quick Start & Installation Guide](#-quick-start--installation-guide)
- [🛠️ Troubleshooting & FAQ](#️-troubleshooting--faq)
- [👤 Author & Acknowledgments](#-author--acknowledgments)

---

## 📌 Executive Summary

Mental health in the modern technology ecosystem is one of the most significant determinants of employee productivity, engineering retention, and sustained innovation. High-stakes sprint cycles, on-call fatigue, imposter syndrome, and cognitive overload frequently lead to unaddressed burnout and attrition.

This repository provides an enterprise-grade, empirical investigation of the **Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey dataset** (1,259 respondents across 27 operational variables). It pairs an **executable 22-chart Jupyter Notebook** structured under the **UBM Framework (Univariate, Bivariate, Multivariate)** with a **futuristic, dark-themed Streamlit Executive Analytics Dashboard** featuring explainable AI risk scoring and an interactive financial ROI calculator.

---

## 🎯 Problem Statement & Business Objectives

### Problem Statement
Tech enterprises invest billions into technical talent acquisition and standard physical healthcare, yet face severe productivity loss and voluntary turnover driven by mental distress. Traditional HR surveys often suffer from social desirability bias and underreporting due to fear of career stagnation or retaliation. Organizations lack actionable, data-driven frameworks to measure the effectiveness of mental wellness benefits, identify disclosure barriers, and quantify the direct return on investment (ROI) of psychological safety initiatives.

### Business Objectives
1. **Empirical Diagnostic**: Quantify the baseline prevalence of mental health treatment demand and identify the strongest statistical predictors of care-seeking behavior.
2. **Stigma & Cultural Auditing**: Evaluate the psychological safety gap between speaking with direct managers versus peers, and measure observed workplace retaliation.
3. **Benefit Optimization**: Identify the "Awareness Gap" in enterprise benefit programs to maximize benefit utilization without inflating operational costs.
4. **Predictive Risk Modeling**: Deploy an explainable Machine Learning model trained with class-imbalance safeguards to simulate individual risk profiles.
5. **Actionable ROI Blueprint**: Deliver an interactive financial model demonstrating how destigmatization and frictionless leave reduce voluntary attrition and save hundreds of thousands in turnover costs.

---

## 📊 Dataset Architecture & Preprocessing Pipeline

### 1. Raw Dataset Overview
- **Source**: Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey
- **Dimensions**: 1,259 rows × 27 features
- **Coverage**: Global tech and non-tech workforce across 48+ countries

### 2. Data Cleaning & Engineering Pipeline
- **Gender Normalization**: Collapsed 40+ raw free-text survey variations into 3 standardized categories: `Male` (78.7%), `Female` (19.6%), and `Non-Binary / LGBTQ+ / Other` (1.7%).
- **Age Outlier Filtering**: Filtered biologically implausible responses (e.g., negative ages, 999 years) to retain verified working professionals aged `18 to 75` (Median = 31.0, IQR = 9.0).
- **Missing Value Handling**:
  - `comments`: Imputed missing values with `"No Comment Provided"` and created a binary indicator `has_comment` (26.6% response rate) for Natural Language Processing (NLP).
  - `work_interfere`: Treated `"Don't Know / N/A"` as distinct informational categories to preserve genuine self-assessment signals.
  - `state`: Retained for geographic analysis in US-based cohorts.

### 3. Data Dictionary Highlights

| Column Name | Data Type | Description & Category Values |
| :--- | :--- | :--- |
| `Age` | Numerical | Age in years (standardized between 18 and 75). |
| `Gender_Clean` | Categorical | Normalized demographic identity (`Male`, `Female`, `Non-Binary / LGBTQ+ / Other`). |
| `Country` | Categorical | Country of employment / residence (Top: US, UK, Canada, Germany, India). |
| `self_employed` | Categorical | Employment status (`Yes`, `No`). |
| `family_history` | Categorical | Family history of diagnosed mental illness (`Yes`, `No`). |
| `treatment` | Binary (Target) | Whether respondent has actively sought treatment for a mental health condition (`Yes`, `No`). |
| `work_interfere` | Categorical | Frequency with which condition impacts work (`Never`, `Rarely`, `Sometimes`, `Often`). |
| `no_employees` | Categorical | Organization size (`1-5`, `6-25`, `26-100`, `100-500`, `500-1000`, `More than 1000`). |
| `remote_work` | Categorical | Works remotely at least 50% of the time (`Yes`, `No`). |
| `tech_company` | Categorical | Primary employer is a technology organization (`Yes`, `No`). |
| `benefits` | Categorical | Employer provides mental health healthcare benefits (`Yes`, `No`, `Don't know`). |
| `care_options` | Categorical | Knowledge of mental health care options provided (`Yes`, `No`, `Not sure`). |
| `wellness_program` | Categorical | Employer has formally discussed mental health (`Yes`, `No`, `Don't know`). |
| `seek_help` | Categorical | Employer provides resources to seek help (`Yes`, `No`, `Don't know`). |
| `anonymity` | Categorical | Anonymity is protected if mental health resources are utilized (`Yes`, `No`, `Don't know`). |
| `leave` | Categorical | Ease of taking medical leave for mental health condition (`Very easy` to `Very difficult`). |
| `mental_health_consequence` | Categorical | Expectation of negative career consequences if disclosed (`Yes`, `No`, `Maybe`). |
| `coworkers` | Categorical | Willingness to discuss mental health with coworkers (`Yes`, `No`, `Some of them`). |
| `supervisor` | Categorical | Willingness to discuss mental health with direct supervisor (`Yes`, `No`, `Some of them`). |
| `mental_health_interview` | Categorical | Willingness to bring up mental health in a job interview (`Yes`, `No`, `Maybe`). |
| `mental_vs_physical` | Categorical | Employer takes mental health as seriously as physical health (`Yes`, `No`, `Don't know`). |
| `obs_consequence` | Categorical | Observed negative consequences for coworkers who disclosed (`Yes`, `No`). |

---

## 🌟 Key Discoveries & Empirical Insights

| Dimension | Key Metric / Empirical Finding | Strategic Organizational Impact |
| :--- | :--- | :--- |
| **Prevalence Baseline** | **50.6% sought treatment**; **61.9% report work interference** (`Sometimes` or `Often`). | Mental health is a majority reality in tech; standardizing wellness coverage is essential for baseline engineering velocity. |
| **Genetic Predisposition** | Family history raises treatment rate from **34.3% to 75.6%** ($p < 10^{-40}$, $\chi^2 = 178.3$). | The single strongest individual predictor of care demand; proves that proactive EAPs directly address existing employee needs. |
| **Gender Disparities** | Women (**68.8%**) and Non-Binary/LGBTQ+ (**80.9%**) seek care at significantly higher rates than Men (**45.3%**). | Highlights masculine emotional concealment and cultural stigma in male-dominated teams; demands targeted destigmatization campaigns. |
| **The Awareness Gap** | In large tech enterprises (>1000 staff), **28.4% do not know** if benefits exist despite **64.5% provision**. | Delivers immediate zero-cost ROI by fixing communication gaps during new-hire onboarding and open enrollment. |
| **Leave Friction vs Safety** | Frictionless medical leave reduces observed workplace retaliation by **75%** (**32.7% down to 8.2%**). | Streamlining mental health leave with zero administrative friction directly reinforces psychological safety. |
| **Manager Leverage** | **40.7%** turn to direct managers first; untrained managers increase retaliation risk **4-fold**. | Training engineering leads in Mental Health First Aid (MHFA) provides the highest organizational leverage. |
| **Interview Taboo** | **79.9%** refuse to discuss mental health in job interviews due to severe hiring discrimination fears. | Establishes the necessity for explicit, public inclusion statements in job descriptions and candidate briefs. |

---

## 🔬 Statistical Hypothesis Testing & Imbalance Auditing

### 1. Structural Class Imbalance Auditing
- **Gender Imbalance**: **78.7% Male**, **19.6% Female**, **1.7% Non-Binary / LGBTQ+**.
- **Industry Imbalance**: **81.9% Tech Companies** vs. **18.1% Non-Tech Employers**.
- **Employment Imbalance**: **88.4% Salaried / W2** vs. **11.6% Self-Employed**.
- **Mitigation Strategy**: Models are trained using **Cost-Sensitive Class Weighting (`class_weight='balanced'`)** and **Stratified K-Fold Cross-Validation** to eliminate algorithmic bias against underrepresented cohorts.

### 2. Chi-Square ($\chi^2$) Hypothesis Testing & Cramér's $V$
Hypothesis testing was conducted against the target variable `treatment` ($\alpha = 0.05$):

| Survey Feature | Degrees of Freedom | Chi-Square ($\chi^2$) | $p$-value | Cramér's $V$ (Effect Size) | Statistical Association |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `work_interfere` | 4 | 294.84 | $1.30 \times 10^{-63}$ | **0.542** | ⭐⭐⭐ Very Strong |
| `family_history` | 1 | 178.27 | $1.16 \times 10^{-40}$ | **0.375** | ⭐⭐⭐ Strong |
| `care_options` | 2 | 94.76 | $2.65 \times 10^{-21}$ | **0.272** | ⭐⭐⭐ Moderate |
| `benefits` | 2 | 64.84 | $8.33 \times 10^{-15}$ | **0.223** | ⭐⭐⭐ Moderate |
| `Gender_Clean` | 2 | 90.13 | $2.23 \times 10^{-04}$ | **0.183** | ⭐⭐ Moderate |
| `leave` | 4 | 29.94 | $5.03 \times 10^{-06}$ | **0.144** | ⭐⭐ Significant |
| `no_employees` | 5 | 24.32 | $1.87 \times 10^{-04}$ | **0.139** | ⭐⭐ Significant |
| `anonymity` | 2 | 16.48 | $2.64 \times 10^{-04}$ | **0.114** | ⭐ Significant |

---

## 🤖 Machine Learning & AI Risk Simulator

### 1. Model Architecture & Benchmarking
Predictive models were trained to evaluate treatment demand and identify the key driver variables:

| Algorithm | Balanced Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Balanced Random Forest** | **74.1%** | **76.2%** | **75.4%** | **0.751** | **0.835** |
| **Cost-Sensitive Logistic Regression** | **73.5%** | **74.9%** | **74.6%** | **0.742** | **0.819** |

### 2. Feature Importance (Mean Decrease in Impurity / Gini)
1. `work_interfere` (34.2% relative importance)
2. `family_history` (18.6% relative importance)
3. `care_options` (11.4% relative importance)
4. `benefits` (9.1% relative importance)
5. `Age` (8.3% relative importance)
6. `Gender_Clean` (7.2% relative importance)

---

## 🖥️ Streamlit Executive Dashboard Guide

The web application (`app.py`) is styled in a **dark, futuristic theme** (`#080c14` to `#0d1527` with `#38bdf8` cyan and `#2dd4bf` emerald accents) and structured into **8 interactive tabs**:

```
+--------------------------------------------------------------------------------------------------+
|                   🧠 MENTAL HEALTH IN TECH | EXECUTIVE DARK ANALYTICS SUITE                      |
+--------------------------------------------------------------------------------------------------+
| [1] 📊 Executive Overview       | Telemetry KPIs, Treatment Donut, Work Interference Gradient    |
| [2] ⚖️ Imbalance & Statistics   | Demographic Skew Pies, Chi-Square Significance Table           |
| [3] 🗺️ Demographics & Geo       | Age Distribution, Gender Care Disparities, Top 10 Countries    |
| [4] 🏢 Culture & Stigma         | Manager vs Peer Comfort, Disclosure Fear, Interview Barrier    |
| [5] 🛡️ Benefits & Infrastructure| Benefits by Size, Leave Friction vs Retaliation, Anonymity     |
| [6] 💬 Employee Voice (NLP)     | Free-Text Keyword Extraction, Sentiment, Verbatim Quotes       |
| [7] 🤖 AI Risk Simulator (ML)   | Balanced Random Forest (0.835 AUC), Gini Importance, Gauge     |
| [8] 💡 Strategic ROI Blueprint  | 4-Pillar Action Roadmap, Interactive Turnover ROI Calculator   |
+--------------------------------------------------------------------------------------------------+
```

### ✨ Advanced Dashboard Features
- **Live Cloud Access**: Hosted on Streamlit Cloud at **[mental-health-tech-analytics.streamlit.app](https://mental-health-tech-analytics.streamlit.app/)**.
- **Dynamic Cohort Filtering**: Filter by Tech vs. Non-Tech, Organization Size, and Country with live sample counter badges (e.g., `United States (751)`, `India (10)`).
- **Intelligent 0-Match Diagnostic Assistant**: Automatically provides context and raw records when highly specific demographic filters return empty slices.
- **NLP Sentiment & Topic Extraction**: Analyzes open-ended comments to surface qualitative themes (stigma, prescription management, burnout).
- **Live AI Gauge**: Interactive profile builder computing real-time treatment probability and personalized intervention recommendations.
- **Interactive ROI Calculator**: Real-time modeling of turnover cost savings based on team size, average engineer salary, and attrition reduction targets.

---

## 💡 Strategic ROI Blueprint & Enterprise Pillars

```
+----------------------------------------------------------------------------------------------------+
|                                 4 STRATEGIC ENTERPRISE PILLARS                                     |
+------------------------------------+---------------------------------------------------------------+
| 1. Bridge the 30% Awareness Gap    | • 1-Click confidential EAP access embedded in Slack & Teams   |
|                                    | • Dedicated 15-minute mental wellness onboarding module       |
+------------------------------------+---------------------------------------------------------------+
| 2. Empathetic Leadership Training  | • Mandatory Mental Health First Aid (MHFA) for Tech Leads     |
|                                    | • Psychological safety KPIs integrated into manager reviews   |
+------------------------------------+---------------------------------------------------------------+
| 3. Frictionless Leave Framework    | • 2 quarterly "no-questions-asked" cognitive recovery days    |
|                                    | • Zero-friction self-service medical leave portal             |
+------------------------------------+---------------------------------------------------------------+
| 4. Inclusive & Tailored Care       | • Specialized provider networks for women & LGBTQ+ engineers  |
|                                    | • Targeted male destigmatization & neurodiversity ERGs        |
+------------------------------------+---------------------------------------------------------------+
```

### 💰 Financial ROI Business Case (Example: 200-Person Engineering Org)
- **Baseline Scenario**: 15% annual voluntary turnover = 30 departures / year.
- **Cost of Engineering Turnover**: $120,000 salary × 50% replacement cost = **$60,000 per lost engineer**.
- **Annual Turnover Cost**: 30 × $60,000 = **$1,800,000**.
- **Targeted Impact**: 20% reduction in mental health-related attrition = **6 engineers retained**.
- **Gross Cost Savings**: 6 × $60,000 = **$360,000 / year**.
- **Total Program Investment**: $200/employee/year EAP + Manager Training = **$45,000 / year**.
- **Net Annual Return**: **$315,000** (**700% ROI**, **<2 Month Payback Period**).

---

## 📁 Repository Structure

```plaintext
Mental Health Survey/
├── survey.csv                             # Raw OSMI dataset (1,259 records, 27 features)
├── Mental_Health_in_Tech_EDA.ipynb        # Complete executed Jupyter Notebook (89 cells, 54 outputs)
├── app.py                                 # Dark-themed interactive Streamlit Web Application
├── generate_and_execute_notebook.py       # Notebook generation and verification automation script
├── requirements.txt                       # Project dependencies and versions
└── README.md                              # Comprehensive project documentation
```

---

## 🚀 Quick Start & Installation Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Number789Alpha/Mental-Health-in-Tech-Survey.git
cd "Mental-Health-in-Tech-Survey"

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Streamlit Analytics Dashboard Locally
```bash
streamlit run app.py
```
The application will launch automatically at `http://localhost:8501`.

### 4. Explore the Jupyter Notebook
```bash
jupyter notebook Mental_Health_in_Tech_EDA.ipynb
```
*(All 89 cells are pre-executed with zero errors and all visualizations embedded).*

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>Q: Why do some small country filters show a sample size warning?</b></summary>
The survey has global reach, but some countries have small cohorts (e.g., India <i>N=10</i>, France <i>N=13</i>). The app includes an intelligent <b>Diagnostic Assistant</b> that surfaces the exact demographics of small cohorts to prevent misleading statistical generalizations.
</details>

<details>
<summary><b>Q: How are non-binary and LGBTQ+ responses handled?</b></summary>
Raw text entries with non-binary, genderqueer, trans, or fluid descriptors were grouped under <code>Non-Binary / LGBTQ+ / Other</code> to ensure their distinct healthcare access patterns are rigorously analyzed rather than discarded.
</details>

<details>
<summary><b>Q: How was the Machine Learning model evaluated for class imbalance?</b></summary>
Models utilize <code>class_weight='balanced'</code> and Stratified K-Fold validation to ensure minority classes are given appropriate cost weighting, achieving a robust <b>0.835 ROC-AUC</b>.
</details>

---

## 👤 Author & Acknowledgments

- **Author**: **Priyank Mishra** ([@Number789Alpha](https://github.com/Number789Alpha))
- **Dataset Source**: [Open Sourcing Mental Illness (OSMI)](https://osmihelp.org/)
- **Analysis Framework**: UBM (Univariate, Bivariate, Multivariate) Industry Standard
- **Live Application**: [https://mental-health-tech-analytics.streamlit.app/](https://mental-health-tech-analytics.streamlit.app/)

---

*Empowering technology leaders to build sustainable, high-velocity engineering organizations anchored in psychological safety, empathy, and data-driven wellness.*
