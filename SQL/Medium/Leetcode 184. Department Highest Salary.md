# SQL Challenge: Leetcode 184. Department Highest Salary

## Description and Goal

Write a solution to find employees who have the highest salary in each department.

## Tables

**Table: Employee**

| Column Name  | Type    |
|--------------|---------|
| id           | int     |
| name         | varchar |
| salary       | int     |
| departmentId | int     |

**id** is the primary key for this table, and **departmentId** is a foreign key that references the `id` from the `Department` table. Each row indicates the ID, name, salary, and department ID of an employee.

**Table: Department**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |

**id** is the primary key for this table. Each row indicates the ID and name of a department.

## Example Input and Output

### Example 1:

**Input:**

Employee table:
| id  | name  | salary | departmentId |
|-----|-------|--------|--------------|
| 1   | Joe   | 70000  | 1            |
| 2   | Jim   | 90000  | 1            |
| 3   | Henry | 80000  | 2            |
| 4   | Sam   | 60000  | 2            |
| 5   | Max   | 90000  | 1            |

Department table:
| id  | name  |
|-----|-------|
| 1   | IT    |
| 2   | Sales |

**Output:**

| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Jim      | 90000  |
| Sales      | Henry    | 80000  |
| IT         | Max      | 90000  |

**Explanation:**
Max and Jim both have the highest salary in the IT department, and Henry has the highest salary in the Sales department.

## Thoughts and Solution Approach

### English:
To solve this problem, we need to find the maximum salary in each department and identify the employees who earn this salary.
- **Join Operations**: Join the `Employee` table with the `Department` table to correlate employees with their departments.
- **Subquery for Maximum Salaries**: Use a subquery in the WHERE clause that selects the maximum salary grouped by department from the `Employee` table.
- **Matching Salaries and Departments**: Ensure that the employee's salary and department match those from the subquery.
- **Last Point**: If multiple employees in the same department have the same salary and it is the highest in that department, they will all be displayed.

### 中文:
解決這個問題，我們需要找出每個部門最高的薪水並找出這筆薪水屬於哪個員工。
- **連接操作**：將 `Employee` 表與 `Department` 表連接，將員工與其所在部門相關聯。
- **最高薪水的子查詢**：在 WHERE 子句中使用一個子查詢，從 `Employee` 表中選擇按部門分組的最大薪水。
- **匹配薪水和部門**：確保員工的薪水和部門與子查詢中的相匹配，但如果薪水相同又是該部門最高的也會同時顯示出來。

## SQL Code

```sql
SELECT 
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM
    Employee e
    JOIN Department d ON e.departmentId = d.id
WHERE
    (e.salary, e.departmentId) IN (
        SELECT MAX(salary), departmentId 
        FROM Employee 
        GROUP BY departmentId
    );
