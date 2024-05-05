# SQL Challenge: Leetcode 183. Customers Who Never Order

## Description and Goal

Write a solution to identify all customers who have never placed an order from the `Customers` and `Orders` tables.

## Tables

**Table: Customers**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |

**id** is the primary key for this table, indicating the ID and name of a customer.

**Table: Orders**

| Column Name | Type |
|-------------|------|
| id          | int  |
| customerId  | int  |

**id** is the primary key for this table. **customerId** is a foreign key that references the `id` from the `Customers` table. Each row indicates the ID of an order and the ID of the customer who placed it.

## Example Input and Output

### Example 1:

**Input:**

Customers table:
| id  | name  |
|-----|-------|
| 1   | Joe   |
| 2   | Henry |
| 3   | Sam   |
| 4   | Max   |

Orders table:
| id  | customerId |
|-----|------------|
| 1   | 3          |
| 2   | 1          |

**Output:**

| Customers |
|-----------|
| Henry     |
| Max       |

## Thoughts and Solution Approach

### English:

The goal is to find customers who do not have any orders associated with them in the `Orders` table:
- **LEFT JOIN Operation**: Perform a `LEFT JOIN` between the `Customers` table and the `Orders` table on the `id` field of `Customers` and the `customerId` field of `Orders`.
- **Identifying Missing Orders**: Use the `WHERE` clause to filter out customers whose `customerId` does not appear in the `Orders` table (i.e., `customerId IS NULL`).

### 中文:

目標是找出在 `Orders` 表中沒有任何訂單的客戶：
- **LEFT JOIN**：在 `Customers` 表和 `Orders` 表之間執行 `LEFT JOIN`，連接 `Customers` 表的 `id` 字段和 `Orders` 表的 `customerId` 字段，使用`LEFT JOIN`能完整保留`Customers`所有紀錄，如果左表紀錄右表沒有，那右表部分將返回`NULL`。
- **識別未下訂單的客戶**：使用 `WHERE` 子句篩選出在 `Orders` 表中沒有出現的客戶（即 `customerId IS NULL`）。

## SQL Code

```sql
SELECT c.name AS Customers
FROM Customers c
LEFT JOIN Orders o ON c.id = o.customerId
WHERE o.customerId IS NULL;
