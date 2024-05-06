# SQL Challenge: Leetcode 1075. Project Employees I

## Description and Goal

Write a SQL query that calculates the average years of experience of all the employees for each project, with the average rounded to two decimal places. This involves joining two tables and using aggregate functions to compute the mean experience per project.

## Tables

**Table: Project**

| Column Name  | Type |
|--------------|------|
| project_id   | int  |
| employee_id  | int  |

**Table: Employee**

| Column Name      | Type     |
|------------------|----------|
| employee_id      | int      |
| name             | varchar  |
| experience_years | int      |

## Example Input and Output

### Example 1:

**Input:**

Project table:
| project_id  | employee_id |
|-------------|-------------|
| 1           | 1           |
| 1           | 2           |
| 1           | 3           |
| 2           | 1           |
| 2           | 4           |

Employee table:
| employee_id | name   | experience_years |
|-------------|--------|------------------|
| 1           | Khaled | 3                |
| 2           | Ali    | 2                |
| 3           | John   | 1                |
| 4           | Doe    | 2                |

**Output:**

| project_id  | average_years |
|-------------|---------------|
| 1           | 2.00          |
| 2           | 2.50          |

**Explanation:**
- The average experience years for project 1 is calculated as (3 + 2 + 1) / 3 = 2.00.
- The average for project 2 is calculated as (3 + 2) / 2 = 2.50.

## Thoughts and Solution Approach

### English:

To achieve the task, need the following steps:
- **Joining Tables**: The `Project` table is joined with the `Employee` table using a `LEFT JOIN` on `employee_id` to ensure all project assignments are matched with employee details.
- **Calculating Average**: The `AVG()` function is used to calculate the average experience years for each project. The issue ask to round to 2 digits.
- **Grouping Data**: Group by `project_id` to ensure that the average is calculated separately for each project.

### 中文:

為了完成這項任務，需要進行下列步驟：
- **表格連接**：使用 `LEFT JOIN` 在 `employee_id` 上將 `Project` 表與 `Employee` 表連接，確保所有的項目分配都與員工詳情匹配。
- **計算平均值**：使用 `AVG()` 函數計算每個項目的平均經驗年數。題目說明需四捨五入到兩位小數。
- **數據分組**：按 `project_id` 分組，確保分別為每個項目計算平均值。

## SQL Code

```sql
SELECT p.project_id, ROUND(AVG(e.experience_years), 2) AS average_years
FROM Project p
LEFT JOIN Employee e ON p.employee_id = e.employee_id
GROUP BY p.project_id;
