# SQL Challenge: Leetcode 577. Employee Bonus

## Description and Goal

Write a solution to report the name and bonus amount of each employee who has a bonus less than 1000. For employees without a bonus entry in the bonus table, they should be included with a bonus value of `null`.

## Tables

**Table: Employee**

| Column Name | Type    |
|-------------|---------|
| empId       | int     |
| name        | varchar |
| supervisor  | int     |
| salary      | int     |

**empId** is the primary key for this table. Each row includes the employee's ID, name, supervisor's ID, and salary.

**Table: Bonus**

| Column Name | Type |
|-------------|------|
| empId       | int  |
| bonus       | int  |

**empId** is the primary key for this table and is a foreign key referencing `empId` from the Employee table. Each row contains the employee's ID and their respective bonus.

## Example Input and Output

### Example 1:

**Input:**

Employee table:
| empId | name   | supervisor | salary |
|-------|--------|------------|--------|
| 3     | Brad   | null       | 4000   |
| 1     | John   | 3          | 1000   |
| 2     | Dan    | 3          | 2000   |
| 4     | Thomas | 3          | 4000   |

Bonus table:
| empId | bonus |
|-------|-------|
| 2     | 500   |
| 4     | 2000  |

**Output:**

| name | bonus |
|------|-------|
| Brad | null  |
| John | null  |
| Dan  | 500   |

**Explanation:**
- Brad and John do not have entries in the bonus table, so they appear with `null` bonuses.
- Dan has a bonus of 500, which is under 1000.
- Thomas is not listed because his bonus is 2000.

## Thoughts and Solution Approach

### English:

The challenge requires joining two tables to get each employee's bonus and filtering the results based on the bonus amount.
- **LEFT JOIN Usage**: Use a `LEFT JOIN` to combine the `Employee` and `Bonus` tables. This ensures all employees are included in the result, even if they do not have a corresponding entry in the `Bonus` table.
- **Conditional Filtering**: The `WHERE` clause filters out employees whose bonuses are either non-existent (`IS NULL`) or less than 1000.

### 中文:

這個挑戰要求連接兩個表來獲取每位員工的獎金，並基於獎金數額來過濾結果。
- **LEFT JOIN 使用**：使用 `LEFT JOIN` 來組合 `Employee` 和 `Bonus` 表。這確保了即使某些員工在 `Bonus` 表中沒有對應的項目，也會包括在結果中。
- **條件過濾**：`WHERE` 子句過濾出那些獎金不存在（`IS NULL`）或少於 1000 的員工。

## SQL Code

```sql
SELECT e.name, b.bonus 
FROM Employee e
LEFT JOIN Bonus b ON e.empId = b.empId
WHERE b.bonus IS NULL OR b.bonus < 1000;
