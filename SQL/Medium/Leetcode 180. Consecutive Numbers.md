# SQL Challenge: Leetcode 180. Consecutive Numbers

## Description and Goal

Find all numbers that appear at least three times consecutively in the `Logs` table and return the result table in any order.

## Tables

**Table: Logs**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| num         | varchar |

**id** is the primary key for this table and is an autoincrement column.

## Example Input and Output

### Example 1:

**Input:**

Logs table:
| id  | num |
|-----|-----|
| 1   | 1   |
| 2   | 1   |
| 3   | 1   |
| 4   | 2   |
| 5   | 1   |
| 6   | 2   |
| 7   | 2   |

**Output:**

| ConsecutiveNums |
|-----------------|
| 1               |

**Explanation:**
1 is the only number that appears consecutively for at least three times.

## Thoughts and Solution Approach

### English:
* To identify numbers that appear consecutively three times in the Logs table, we can utilize a SQL Common Table Expression (CTE) to create a temporary result set
  that we can then reference within our main SQL query.

 - **CTE Construction**:

    - **Create the CTE**: First, Define a CTE named cte that selects the num column and the next two consecutive num values using the LEAD function. The LEAD function is part of SQL's window functions that provide a way to access data from subsequent rows without using a self-join.
      
    - **Specify Ordering**: It's crucial to order the rows to ensure the LEAD function accurately fetches the next values. In most cases, this would typically be done by an id or a timestamp column to reflect the true sequence of events or entries.
      
    - **Apply the LEAD Function**: Within the CTE, we apply the LEAD function twice to get the next two consecutive numbers for each row based on the specified order.
      
 - **Final Query Using the CTE**:

    - **Filter Results**: After defining the CTE, the main query filters out the rows where the current number and its next two consecutive numbers are the same.
      
    - **Select Distinct Numbers**: Finally, we select the distinct numbers that meet the criteria of appearing three consecutive times, ensuring that each number is listed only once in the result set.

### 中文:

* 要找出在 Logs 表中連續出現至少三次的數字，我們可以使用 SQL 的公用表達式（CTE）來建立一個暫時的結果集，然後在主查詢中引用這個結果集。

  - **創建 CTE**：首先，定義一個名為 cte 的 CTE，它選取 num 欄位以及使用 LEAD 函數獲得的接下來兩個連續的 num 值。LEAD 函數是窗口函數的一部分，它允許我們在不使用自關聯的情況下訪問後續行的數據。
    
  - **指定排序**：確保行的排序是必要的，以保證 LEAD 函數能夠準確地獲取下一個值。通常情況下，這通常是通過 id 或時間戳列來完成的，以反映事件或條目的真實順序。
    
  - **應用 LEAD 函數**：在 CTE 中，應用 LEAD 函數兩次，根據指定的順序獲得每一行的下兩個連續數字。

- **使用 CTE 的最終查詢**：

  - **篩選結果**：在定義了 CTE 之後，主查詢過濾出當前數字及其下兩個連續數字相同的行。
    
  - **選擇獨特的數字**：最後，選擇符合連續出現三次的標準的獨特數字，確保每個數字在結果集中只列出一次。

## SQL Code

```sql
WITH cte AS (
    SELECT num,
           LEAD(num, 1) OVER (ORDER BY id) AS num1,
           LEAD(num, 2) OVER (ORDER BY id) AS num2
    FROM Logs
)
SELECT DISTINCT num AS ConsecutiveNums
FROM cte
WHERE num = num1 AND num = num2;
