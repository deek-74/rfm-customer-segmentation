-- Monthly KPIs from staged fact sales (product net revenue).
-- Expects registered table: fact_sales

WITH product AS (
    SELECT *
    FROM fact_sales
    WHERE line_type = 'sale'
      AND net_revenue > 0
)
SELECT
    month_start,
    MAX(CAST(is_partial_month AS INTEGER)) = 1 AS is_partial_month,
    SUM(net_revenue) AS net_revenue,
    COUNT(DISTINCT CASE WHEN is_countable_order THEN InvoiceNo END) AS orders,
    COUNT(DISTINCT CustomerID) AS customers_active,
    COUNT(DISTINCT CASE WHEN CustomerID IS NOT NULL THEN CustomerID END) AS identified_customers_active,
    SUM(CASE WHEN CustomerID IS NULL THEN net_revenue ELSE 0 END) AS guest_net_revenue,
    SUM(CASE WHEN CustomerID IS NOT NULL THEN net_revenue ELSE 0 END) AS identified_net_revenue
FROM product
GROUP BY month_start
ORDER BY month_start
