# SQL Challenge: Leetcode 185. Department Top Three Salaries

## Description and Goal

Write a solution to identify employees who are high earners within their respective departments. A high earner is defined as an employee whose salary is among the top three unique salaries in their department.

## Tables

**Table: Employee**

| Column Name  | Type    |
|--------------|---------|
| id           | int     |
| name         | varchar |
| salary       | int     |
| departmentId | int     |

**id** is the primary key for this table, and **departmentId** is a foreign key referencing the `id` from the `Department` table. Each row shows the ID, name, salary, and department ID of an employee.

**Table: Department**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |

**id** is the primary key for this table. Each row shows the ID and name of a department.

## Example Input and Output

### Example 1:

**Input:**

Employee table:
| id  | name  | salary | departmentId |
|-----|-------|--------|--------------|
| 1   | Joe   | 85000  | 1            |
| 2   | Henry | 80000  | 2            |
| 3   | Sam   | 60000  | 2            |
| 4   | Max   | 90000  | 1            |
| 5   | Janet | 69000  | 1            |
| 6   | Randy | 85000  | 1            |
| 7   | Will  | 70000  | 1            |

Department table:
| id  | name  |
|-----|-------|
| 1   | IT    |
| 2   | Sales |

**Output:**

| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Max      | 90000  |
| IT         | Joe      | 85000  |
| IT         | Randy    | 85000  |
| IT         | Will     | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |

**Explanation:**
- In the IT department, Max earns the highest unique salary, Joe and Randy the second-highest unique salary, and Will the third-highest unique salary.
- In the Sales department, Henry earns the highest salary and Sam the second-highest salary, with no third highest as only two employees exist.

## Thoughts and Solution Approach

### English:

To tackle this problem, the approach involves:
- **Using Window Functions**:
   - **`DENSE_RANK()`**: This function is used to assign a rank to each employee based on their salary within their department. The ranking is done in descending order of salary.
   - **Partitioning**: The `PARTITION BY` clause in the window function ensures that the ranking resets for each department. That is, the ranking starts anew for each department.
   - **Ordering**: Within each partition (department), employees are ordered by their salary in descending order. This means that the highest salary in each department receives a rank of 1, the second highest (if different) receives a rank of 2, and so forth.

- **Defining Columns with Aliases**:
   - **Department**: The name of the department is retrieved from the `Department` table and displayed as `Department`.
   - **Employee**: Similarly, the employee's name is listed under the alias `Employee`.
   - **Salary**: The salary is shown under the column `Salary`.

- **Filtering Top Earners**:
   - After applying the window function, a subquery is used where we select only those rows where the employee's rank is 3 or less (`D_rank <= 3`). This step effectively filters out all but the top three earners in each department.


### 中文（繁體）解題思路:

要解決這個問題，方法包括：
- **窗口函數使用**：使用 `DENSE_RANK()` 窗口函數對每個部門內的員工按薪水降序進行排名。
- **按部門分區**：窗口函數中的 `PARTITION BY` 子句確保每個部門的排名是獨立的，即每個部門的排名都是從新開始的。
- **過濾高收入者**：在每個分區（部門）內，員工按薪水降序排序。這意味著每個部門中最高的薪水獲得第一名，第二高的（如果不同）獲得第二名，依此類推。

## SQL Code

```sql
SELECT Department, Employee, Salary
FROM (
    SELECT
        d.name AS Department,
        e.name AS Employee,
        e.salary AS Salary,
        DENSE_RANK() OVER (PARTITION BY d.name ORDER BY e.salary DESC) AS D_rank
    FROM Employee e
    JOIN Department d ON e.departmentId = d.id    
) AS rank_row
WHERE D_rank <= 3;
