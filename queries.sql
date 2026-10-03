
-- Query 1: Total Net Revenue
SELECT
    SUM(quantity * unit_price * (1 - discount)) AS total_net_revenue
FROM orders;

-- Query 2: Top 10 Customers by Spending
SELECT
    c.customer_name,
    c.city,
    COUNT(o.order_id) AS number_of_orders,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS total_spending
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
GROUP BY c.customer_id
ORDER BY total_spending DESC
LIMIT 10;

-- Query 3: Category-wise Revenue, Orders and Quantity Sold
SELECT
    p.category,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS revenue,
    COUNT(o.order_id) AS order_count,
    SUM(o.quantity) AS quantity_sold
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
GROUP BY p.category
ORDER BY revenue DESC;

-- Query 4: Monthly Net Revenue Trend
SELECT
    strftime('%Y-%m', order_date) AS month,
    SUM(quantity * unit_price * (1 - discount)) AS monthly_revenue
FROM orders
GROUP BY month
ORDER BY month;

-- Query 5: Top 5 Products by Net Revenue
SELECT
    p.product_name,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS revenue
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
GROUP BY p.product_id
ORDER BY revenue DESC
LIMIT 5;
