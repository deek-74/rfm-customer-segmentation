RFM Customer Segmentation

Product analytics using the UCI Online Retail dataset (UK giftware retailer, Dec 2010–9 Dec 2011) to identify who to protect, who is slipping, and where win-back campaigns are worth pursuing. All revenue figures represent historical net revenue, not forecasts.

Key Results
4,324 customers · £8.25M historical net revenue · £452 AOV · 65.3% repeat purchase rate
Champions: 18% of customers → 63% of revenue
Top 20%: 74% of revenue (Gini 0.70)
At-Risk + Can't Lose Them: 400 customers · £557K revenue
82% of June Champions remained Champions in December
December 2011 covers 1–9 Dec only and should not be interpreted as a full-month decline

Methodology
Net cancellations using LIFO and exclude postage/manual charges from product revenue
Calculate customer-level RFM metrics using DuckDB SQL
Apply tie-safe R/M quintiles and fixed Frequency bands: 1 / 2 / 3–4 / 5–9 / 10+
Assign customers to 10 covering segments using first-match-wins logic
Apply K-means (k=3) on log1p-transformed RFM metrics, with cluster names based on median profiles
Analyse June–December segment migration

Tech Stack

Python · pandas · DuckDB · SQL · scikit-learn · XlsxWriter · Plotly

Deliverable

outputs/RFM_Customer_Segmentation_Workbook.xlsx — Excel workbook containing the executive summary, charts, RFM rules, customer-level analysis, segment summaries, and transaction-level audit sheets.
