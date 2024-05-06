# SQL Challenge: Leetcode 1251. Average Selling Price

## Description and Goal

Write a solution to calculate the average selling price for each product based on its sales and varying prices over different periods. The average selling price should consider the total revenue generated from a product and the total units sold, ensuring accurate pricing information that reflects discounts or price changes over time.

## Tables

**Table: Prices**

| Column Name  | Type |
|--------------|------|
| product_id   | int  |
| start_date   | date |
| end_date     | date |
| price        | int  |

- **product_id, start_date, end_date**: Composite primary key that uniquely identifies a pricing period for a product.
- **price**: Price of the product during the specified period.

**Table: UnitsSold**

| Column Name   | Type |
|---------------|------|
| product_id    | int  |
| purchase_date | date |
| units         | int  |

- Each row indicates the units of a product sold on a specific date.

## Example Input and Output

### Example 1:

**Input:**

Prices table:
| product_id | start_date | end_date   | price |
|------------|------------|------------|-------|
| 1          | 2019-02-17 | 2019-02-28 | 5     |
| 1          | 2019-03-01 | 2019-03-22 | 20    |
| 2          | 2019-02-01 | 2019-02-20 | 15    |
| 2          | 2019-02-21 | 2019-03-31 | 30    |

UnitsSold table:
| product_id | purchase_date | units |
|------------|---------------|-------|
| 1          | 2019-02-25    | 100   |
| 1          | 2019-03-01    | 15    |
| 2          | 2019-02-10    | 200   |
| 2          | 2019-03-22    | 30    |

**Output:**

| product_id | average_price |
|------------|---------------|
| 1          | 6.96          |
| 2          | 16.96         |

**Explanation:**
- For product 1, the average selling price = ((100 units * $5) + (15 units * $20)) / 115 units = $6.96.
- For product 2, the average selling price = ((200 units * $15) + (30 units * $30)) / 230 units = $16.96.

## Thoughts and Solution Approach

### English:

To find the average selling price:
- **Joining Tables**: Use a `LEFT JOIN` to connect the `Prices` and `UnitsSold` tables on `product_id` while ensuring the sale date falls within the pricing period defined by `start_date` and `end_date`.
- To accurately calculate the average selling price for each product, the SQL query must ensure that the sales data (units sold on specific dates) matches the correct pricing data for those dates.
- This is achieved by using the `BETWEEN` condition in the query. This condition filters the sales data to only include sales that occurred during the valid pricing period defined in the `Prices` table (`start_date` to `end_date`).
- This approach ensures that each sale is multiplied by the correct price, thus providing an accurate calculation of the average selling price.
- **Weighted Average Calculation**: Calculate the total revenue for each product by multiplying the units sold by the price at which they were sold, then divide by the total units sold to find the average price.
- **Handling Missing Data**: Use `IFNULL` to handle cases where there are no sales (ensuring the result defaults to 0 if there are no matching rows after the join).

### 中文:

要計算每個產品的平均銷售價格：
- **表格連接**：使用 `LEFT JOIN` 將 `Prices` 表和 `UnitsSold` 表連接在一起，並確保銷售日期落在由 `start_date` 和 `end_date` 定義的價格時期內。
- 為了精確計算每個產品的平均售價，SQL 查詢必須確保銷售數據（特定日期售出的單位）與那些日期的正確價格數據相匹配，透過使用`BETWEEN` 來實現。
- 此條件過濾銷售數據，只包括在 `Prices` 表中定義的有效定價期間（`start_date` 至 `end_date`）內發生的銷售。這種方法確保每筆銷售都按正確的價格計算，從而提供了平均售價的精確計算。
- **加權平均計算**：通過將售出的單位與其售出的價格相乘來計算每個產品的總收入，然後除以售出的總單位來找出平均價格。
- **處理缺失數據**：使用 `IFNULL` 處理沒有銷售的情況（確保在連接後沒有匹配時結果默認為0）。

## SQL Code

```sql
SELECT p.product_id, IFNULL(ROUND(SUM(u.units * p.price) / SUM(u.units), 2), 0) AS average_price
FROM Prices p
LEFT JOIN UnitsSold u ON p.product_id = u.product_id
AND u.purchase_date BETWEEN p.start_date AND p.end_date
GROUP BY p.product_id;
