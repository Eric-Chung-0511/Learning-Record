# SQL Challenge: Leetcode 1633. Percentage of Users Attended a Contest

## Description and Goal

Write an SQL query to find the percentage of users registered for each contest, rounded to two decimal places. The percentage should reflect how many of the total users have registered for each contest, with results ordered by percentage in descending order and, in case of ties, by contest_id in ascending order.

## Tables

**Table: Users**

| Column Name | Type    |
|-------------|---------|
| user_id     | int     |
| user_name   | varchar |

**Table: Register**

| Column Name | Type |
|-------------|------|
| contest_id  | int  |
| user_id     | int  |

## Example Input and Output

### Example 1:

**Input:**

Users table:
| user_id | user_name |
|---------|-----------|
| 6       | Alice     |
| 2       | Bob       |
| 7       | Alex      |

Register table:
| contest_id | user_id |
|------------|---------|
| 215        | 6       |
| 209        | 2       |
| ...        | ...     |

**Output:**

| contest_id | percentage |
|------------|------------|
| 208        | 100.0      |
| 209        | 100.0      |
| ...        | ...        |

**Explanation:**
Contests 208, 209, and 210 had all users registered, thus 100%. Contest 215 had two out of three users registered, resulting in 66.67%.

## Thoughts and Solution Approach

### English:

The issue aims to calculate the registration percentage by:
- **Calculating Unique Registrants**: `count(distinct user_id)` ensures that each user is only counted once per contest.
- **Total User Base**: `(select count(user_id) from users)` computes the total number of users from the `Users` table, serving as the denominator.
- **Percentage Calculation**: Multiplying the count of distinct users by 100 before dividing by the total user count helps prevent precision loss that can occur from dividing small numbers.
- **Sorting**: Results are ordered primarily by `percentage` in descending order and secondarily by `contest_id` in ascending order in case of ties.

### 中文:

此題目主旨在於通過以下步驟計算每個競賽的註冊百分比：
- **計算獨特的註冊者數量**：`count(distinct user_id)` 確保每個用戶在每個競賽中只被計數一次。
- **總用戶基數**：`(select count(user_id) from users)` 從 `Users` 表中計算總用戶數，作為分母。
- **百分比計算**：在除以總用戶數前將獨特用戶數乘以 100，有助於防止由於分割小數而導致的精度損失。也可以全部除完再*100，但因為有小數，所以會造成精度損失。
- **排序**：結果首先按 `percentage` 降序排序，在百分比相同的情況下按 `contest_id` 升序排序。

## SQL Code

```sql
SELECT contest_id, ROUND(COUNT(DISTINCT user_id) * 100 / (SELECT COUNT(user_id) FROM Users), 2) AS percentage
FROM Register
GROUP BY contest_id
ORDER BY percentage DESC, contest_id;
