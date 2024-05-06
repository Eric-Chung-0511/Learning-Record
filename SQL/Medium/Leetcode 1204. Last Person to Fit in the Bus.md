# SQL Challenge: Leetcode 1204. Last Person to Fit in the Bus

## Description and Goal

Write a SQL query to find the name of the last person who can board the bus without exceeding the weight limit of 1000 kilograms. It is guaranteed that the first person does not exceed the weight limit.

## Tables

**Table: Queue**

| Column Name  | Type    |
|--------------|---------|
| person_id    | int     |
| person_name  | varchar |
| weight       | int     |
| turn         | int     |

## Example Input and Output

### Example 1:

**Input:**

Queue table:
| person_id | person_name | weight | turn |
|-----------|-------------|--------|------|
| 5         | Alice       | 250    | 1    |
| 3         | Alex        | 350    | 2    |
| 6         | John Cena   | 400    | 3    |
| 2         | Marie       | 200    | 4    |
| 4         | Bob         | 175    | 5    |
| 1         | Winston     | 500    | 6    |

**Output:**

| person_name |
|-------------|
| John Cena   |

**Explanation:**
The table is ordered by the turn for simplicity:
+------+----+-----------+--------+--------------+
| Turn | ID | Name      | Weight | Total Weight |
+------+----+-----------+--------+--------------+
| 1    | 5  | Alice     | 250    | 250          |
| 2    | 3  | Alex      | 350    | 600          |
| 3    | 6  | John Cena | 400    | 1000         | (last person to board)
| 4    | 2  | Marie     | 200    | 1200         | (cannot board)

## Thoughts and Solution Approach

### English:

The SQL query identifies the last person who can board by:
- **Joining Table with Itself**: The queue table is self-joined on the condition q1.turn >= q2.turn. This allows the accumulation of the total weight each person would contribute if they boarded the bus in the order of their turn.
- **Calculating Total Weight**: For each person determined by their turn, the sum of weights of all persons who would have boarded before or at the same time is calculated.
- **Filtering and Sorting**: The result is filtered to only include those whose cumulative weight is within the limit, then ordered by total weight in descending order to find the maximum total weight under or equal to 1000.
- **Limiting Results**: The query returns only the top result after ordering, which corresponds to the last person who can board without exceeding the limit.

### 中文:

這個 SQL 查詢透過以下步驟來確定最後一個可以登車的人：
- **表自身關聯**: `queue` 表根據條件 `q1.turn >= q2.turn` 與自身進行關聯，這使得每個人如果按照他們的輪次上車的話，可以計算累計重量。。
- **計算總重量**: 對於每個由其輪次(`q1.turn`) 確定的人，計算所有在此人之前或同時上車的人的重量總和。
- **過濾與排序**: 結果被過濾，只包括那些累積重量在限制內的記錄，然後按總重量降序排序，以找到小於或等於 1000 的最大總重量。
- **限制結果**: 查詢在排序後只返回最上面的一條結果，這對應於最後一個不超過限重上車的人。

## SQL Code

```sql
SELECT q1.person_name
FROM queue q1
JOIN queue q2 ON q1.turn >= q2.turn
GROUP BY q1.turn
HAVING SUM(q2.weight) <= 1000
ORDER BY SUM(q2.weight) DESC
LIMIT 1;
