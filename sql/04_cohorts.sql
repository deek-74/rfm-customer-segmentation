-- Cohort activity: one row per customer per calendar month with a countable order.
-- Expects registered tables: dim_customer, fact_sales
-- Partial months are flagged so Python can drop them from denominators.

WITH orders AS (
    SELECT DISTINCT
        CustomerID,
        CAST(month_start AS DATE) AS activity_month,
        MAX(CAST(is_partial_month AS INTEGER)) = 1 AS is_partial_month,
        SUM(net_revenue) AS month_revenue
    FROM fact_sales
    WHERE CustomerID IS NOT NULL
      AND is_countable_order
      AND line_type = 'sale'
      AND net_revenue > 0
    GROUP BY CustomerID, CAST(month_start AS DATE)
),
cohorts AS (
    SELECT
        CustomerID,
        DATE_TRUNC('month', first_purchase_date)::DATE AS cohort_month,
        first_purchase_date
    FROM dim_customer
)
SELECT
    c.CustomerID,
    c.cohort_month,
    o.activity_month,
    DATE_DIFF('month', c.cohort_month, o.activity_month) AS months_since,
    o.month_revenue,
    o.is_partial_month
FROM cohorts AS c
INNER JOIN orders AS o
    ON c.CustomerID = o.CustomerID
ORDER BY c.cohort_month, o.activity_month, c.CustomerID
