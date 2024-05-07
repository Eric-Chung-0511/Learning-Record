# SQL Challenge: Leetcode 176.Find the Second Highest Salary

## Description and Goal

Write a solution to find the second highest salary from the Employee table. If there is no second highest salary, the result should return null (or None in some programming environments).

## Tables

**Table: Employee**

| Column Name | Type |
|-------------|------|
| id          | int  |
| salary      | int  |

**id** is the primary key (column with unique values) for this table. Each row of this table contains information about the salary of an employee.

## Example Input and Output

### Example 1:

**Input:**

Employee table:
| id | salary |
|----|--------|
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |

**Output:**

| SecondHighestSalary |
|---------------------|
| 200                 |

### Example 2:

**Input:**

Employee table:
| id | salary |
|----|--------|
| 1  | 100    |

**Output:**

| SecondHighestSalary |
|---------------------|
| null                |

## Thoughts and Solution Approach

### English
* To solve this problem, we need to identify the second highest unique salary from the `Employee` table. 
* A common approach is to use SQL window functions or a subquery that filters out the highest salary and then finds the maximum of the remaining salaries. 
* If there is no second highest salary available, the query should be designed to return `NULL`.

### 中文
* 要先找到 `Employee` 表中第二高的唯一薪水
* 常見的方法可以用窗口函數或子查詢，這邊使用子查詢先找到最高薪水，然後從剩餘薪水值中找到最大值，也就是第二高的薪水。
* 如果表中沒有第二高的薪水，例如只有一個員工或所有員工薪水相同，則返回 `NULL` 。

## SQL Code

```sql
SELECT MAX(salary) AS SecondHighestSalary
FROM Employee
WHERE salary < (
    SELECT MAX(salary) FROM Employee
);

