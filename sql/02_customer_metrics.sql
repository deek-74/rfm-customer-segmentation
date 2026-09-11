-- One row per identified customer with positive historical net revenue.
-- Expects registered table: fact_sales
-- Parameters: {{snapshot_date}}  (YYYY-MM-DD, max invoice date normalised + 1 day)

WITH product_sales AS (
    SELECT *
    FROM fact_sales
    WHERE CustomerID IS NOT NULL
      AND line_type = 'sale'
      AND net_revenue > 0
),
customer_value AS (
    SELECT
        CustomerID,
        SUM(net_revenue) AS monetary
    FROM product_sales
    GROUP BY CustomerID
    HAVING SUM(net_revenue) > 0
),
orders AS (
    SELECT DISTINCT
        CustomerID,
        InvoiceNo,
        CAST(MIN(Date) AS DATE) AS order_date
    FROM fact_sales
    WHERE CustomerID IS NOT NULL
      AND is_countable_order
      AND line_type = 'sale'
      AND net_revenue > 0
    GROUP BY CustomerID, InvoiceNo
),
order_stats AS (
    SELECT
        CustomerID,
        COUNT(*) AS frequency,
        MIN(order_date) AS first_purchase_date,
        MAX(order_date) AS last_purchase_date
    FROM orders
    GROUP BY CustomerID
),
country_counts AS (
    SELECT
        CustomerID,
        Country,
        COUNT(*) AS n_lines
    FROM product_sales
    GROUP BY CustomerID, Country
),
country_picked AS (
    SELECT
        CustomerID,
        Country
    FROM country_counts
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY CustomerID
        ORDER BY n_lines DESC, Country
    ) = 1
),
product_counts AS (
    SELECT
        CustomerID,
        COUNT(DISTINCT StockCode) AS distinct_products
    FROM product_sales
    GROUP BY CustomerID
)
SELECT
    v.CustomerID,
    c.Country,
    o.first_purchase_date,
    o.last_purchase_date,
    DATE_DIFF('day', o.last_purchase_date, DATE '{{snapshot_date}}') AS recency_days,
    DATE_DIFF('day', o.first_purchase_date, DATE '{{snapshot_date}}') AS tenure_days,
    o.frequency,
    v.monetary,
    v.monetary / o.frequency AS avg_order_value,
    CASE
        WHEN o.frequency <= 1 THEN NULL
        ELSE DATE_DIFF('day', o.first_purchase_date, o.last_purchase_date) * 1.0
             / (o.frequency - 1)
    END AS avg_days_between_orders,
    p.distinct_products
FROM customer_value AS v
INNER JOIN order_stats AS o
    ON v.CustomerID = o.CustomerID
LEFT JOIN country_picked AS c
    ON v.CustomerID = c.CustomerID
LEFT JOIN product_counts AS p
    ON v.CustomerID = p.CustomerID
ORDER BY v.CustomerID
