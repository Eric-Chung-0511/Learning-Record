# SQL Challenge: Leetcode 181. Employees Earning More Than Their Managers

## Description and Goal

Write a solution to find the employees who earn more than their managers in the `Employee` table.

## Tables

**Table: Employee**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |
| salary      | int     |
| managerId   | int     |

**id** is the primary key for this table. Each row indicates the ID of an employee, their name, salary, and the ID of their manager.

## Example Input and Output

### Example 1:

**Input:**

Employee table:
| id  | name  | salary | managerId |
|-----|-------|--------|-----------|
| 1   | Joe   | 70000  | 3         |
| 2   | Henry | 80000  | 4         |
| 3   | Sam   | 60000  | Null      |
| 4   | Max   | 90000  | Null      |

**Output:**

| Employee |
|----------|
| Joe      |

**Explanation:**
Joe is the only employee who earns more than his manager.

## Thoughts and Solution Approach

### English:

To solve this problem, we need to identify employees who are earning more than their respective managers. This requires comparing each employee's salary with their manager's salary.
- **Self-Join on Employee Table**: We join the `Employee` table to itself to pair each employee with their manager.
- **Comparison Condition**: In the join condition, `managerId` of the first instance of the table (employee) is matched with the `id` of the second instance (manager). The where clause is then used to filter the records where the employee's salary exceeds the manager's salary.

### 中文（繁體）:

解決這個問題，我們需要識別出薪水高於其各自經理的員工。這需要比較每位員工的薪水與其經理的薪水。
- **自我連接員工表**：在連接條件中，`Employee e1` 代表員工，表中 `managerId` 與 `Employee e2` 代表經理，表中 `id` 相匹配。這樣的`JOIN`讓我們可以在同一查詢中比較員工與其經理的薪資。
- **比較條件**：使用 WHERE 子句篩選出員工薪水高於經理薪水的記錄。

## SQL Code

```sql
SELECT e1.name AS Employee
FROM Employee e1
JOIN Employee e2 ON e1.managerId = e2.id
WHERE e1.salary > e2.salary;
