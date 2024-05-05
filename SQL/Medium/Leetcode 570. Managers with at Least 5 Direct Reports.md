# SQL Challenge: Leetcode 570. Managers with at Least 5 Direct Reports

## Description and Goal

Write a solution to identify managers who have at least five direct reports in an organization. This involves counting the number of employees that each manager is responsible for and selecting those where the count is five or more.

## Tables

**Table: Employee**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |
| department  | varchar |
| managerId   | int     |

- **id**: Primary key, unique ID of each employee.
- **name**: Name of the employee.
- **department**: Department in which the employee works.
- **managerId**: ID of the employee's manager. If `null`, the employee does not have a manager.

## Example Input and Output

### Example 1:

**Input:**

Employee table:
| id  | name  | department | managerId |
|-----|-------|------------|-----------|
| 101 | John  | A          | null      |
| 102 | Dan   | A          | 101       |
| 103 | James | A          | 101       |
| 104 | Amy   | A          | 101       |
| 105 | Anne  | A          | 101       |
| 106 | Ron   | B          | 101       |

**Output:**

| name |
|------|
| John |

**Explanation:**
John, identified by ID 101, manages five employees (Dan, James, Amy, Anne, and Ron), meeting the criterion of having at least five direct reports.

## Thoughts and Solution Approach

### English:
To find managers with at least five direct reports, you can:
- **Use a JOIN**: The `Employee` table is joined on itself to associate each employee with their manager.
- **Grouping**: Group the data by `managerId`.
- **Filtering Results and Counting**: Use the `HAVING` and `COUNT` to filter out groups that have five or more entries and count the number of direct reports each manager has.
  
### 中文:
目的是找出至少有五名直接下屬的管理者：
- **使用 JOIN**: 將`Employee`表 `JOIN`，與每位員工與他們的管理者關聯起來。
- **分組**: 按 `managerId` 對數據進行分組。
- **過濾與計算次數**: 使用 `HAVING` 與 `COUNT` 篩選出至少有五名直接下屬的主管。這裡的 `COUNT(*)` 函數計算的是每個 `managerId` 在表中出現的次數，即一位管理者有多少名直接下屬。


## SQL Code
```sql
SELECT e1.name
FROM employee e1 
JOIN employee e2 ON e1.id = e2.managerId
GROUP BY e1.id
HAVING COUNT(*) >= 5;

