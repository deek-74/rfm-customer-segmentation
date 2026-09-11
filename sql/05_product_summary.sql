-- Product ranking on net revenue of product sales only (non-product excluded by line_type).
-- Expects registered table: fact_sales

SELECT
    StockCode,
    ANY_VALUE(Description) AS Description,
    SUM(net_quantity) AS net_quantity,
    SUM(net_revenue) AS net_revenue,
    COUNT(DISTINCT InvoiceNo) AS invoices,
    COUNT(DISTINCT CustomerID) AS customers
FROM fact_sales
WHERE line_type = 'sale'
  AND net_revenue > 0
GROUP BY StockCode
ORDER BY net_revenue DESC
