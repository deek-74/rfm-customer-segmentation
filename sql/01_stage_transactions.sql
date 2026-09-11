-- Stage line-level transactions and flag invoices with positive net revenue.
-- Expects registered table: stg_lines

WITH invoice_net AS (
    SELECT
        InvoiceNo,
        SUM(net_revenue) AS invoice_net_revenue
    FROM stg_lines
    GROUP BY InvoiceNo
)
SELECT
    l.InvoiceNo,
    l.StockCode,
    l.Description,
    l.Quantity,
    l.InvoiceDate,
    l.Date,
    l.month_start,
    l.UnitPrice,
    l.CustomerID,
    l.Country,
    l.line_type,
    l.net_quantity,
    l.net_revenue,
    l.unmatched_qty,
    l.gross_revenue,
    l.is_partial_month,
    (i.invoice_net_revenue > 0) AS is_countable_order,
    i.invoice_net_revenue
FROM stg_lines AS l
LEFT JOIN invoice_net AS i
    ON l.InvoiceNo = i.InvoiceNo
