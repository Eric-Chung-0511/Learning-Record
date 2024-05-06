# SQL Challenge: Leetcode 1174. Immediate Food Delivery II

## Description and Goal

Write a SQL query to find the percentage of immediate orders among the first orders made by all customers. An order is considered "immediate" if the preferred delivery date is the same as the order date. The result should be rounded to two decimal places.

## Tables

**Table: Delivery**

| Column Name                 | Type |
|-----------------------------|------|
| delivery_id                 | int  |
| customer_id                 | int  |
| order_date                  | date |
| customer_pref_delivery_date | date |

## Example Input and Output

### Example 1:

**Input:**

Delivery table:
| delivery_id | customer_id | order_date | customer_pref_delivery_date |
|-------------|-------------|------------|-----------------------------|
| 1           | 1           | 2019-08-01 | 2019-08-02                  |
| 2           | 2           | 2019-08-02 | 2019-08-02                  |
| ...         | ...         | ...        | ...                         |

**Output:**

| immediate_percentage |
|----------------------|
| 50.00                |

**Explanation:**
- Customer 1's first order was scheduled.
- Customer 2's first order was immediate.
- 50% of customers' first orders were immediate.

## Thoughts and Solution Approach

### English:

The approach involves:
- **Identifying First Orders**: By determining the earliest order date for each customer using a subquery that groups by `customer_id` and selects the minimum `order_date`.
- **Immediate Orders Calculation**: Comparing `order_date` with `customer_pref_delivery_date` for these first orders to identify if they are immediate.
- **Percentage Calculation**: Using `AVG()` function where true equals 1 (immediate) and false equals 0 (scheduled), then multiplying by 100 to convert to a percentage.
- **Subquery**: `GROUP BY customer_id` organizes the entries in the Delivery table into groups for each customer. `MIN(order_date)` selects the earliest order date from each group, representing the first order date for each customer.


### 中文:

要完成這查詢需：
- **識別首個訂單**：通過對每個客戶進行分組，選擇最早的 `order_date` 來確定每個客戶的首個訂單。
- **計算即時訂單**：比較這些首個訂單的 `order_date` 與 `customer_pref_delivery_date` 是否相同，以識別它們是否為即時訂單。
- **百分比計算**：使用 `AVG()` ，其中真值等於 1（即時），假值等於 0（預定），然後乘以 100 轉換為百分比。
- **子查詢**: `GROUP BY customer_id` 這裡對 Delivery 表中的每個客戶進行分組， `MIN(order_date)` 從每組中選出最早的訂單日期，即每個客戶的首次訂單日期。

## SQL Code

```sql
SELECT ROUND(AVG(order_date = customer_pref_delivery_date) * 100, 2) AS immediate_percentage
FROM Delivery
WHERE (customer_id, order_date) IN (
    SELECT customer_id, MIN(order_date)
    FROM Delivery
    GROUP BY customer_id
);
