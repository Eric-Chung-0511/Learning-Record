# SQL Challenge: Leetcode 182. Duplicate Emails

## Description and Goal

Write a solution to report all the duplicate emails from the `Person` table. It's guaranteed that the email field is not NULL.

## Tables

**Table: Person**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| email       | varchar |

**id** is the primary key for this table. Each row contains an email, and all emails are in lowercase letters.

## Example Input and Output

### Example 1:

**Input:**

Person table:
| id   | email   |
|------|---------|
| 1    | a@b.com |
| 2    | c@d.com |
| 3    | a@b.com |

**Output:**

| Email   |
|---------|
| a@b.com |

**Explanation:**
a@b.com is repeated two times.

## Thoughts and Solution Approach

### English:

The goal is to identify duplicate emails in the `Person` table. Given that each email field is non-null and the table includes a primary key (`id`), we can efficiently group the records by the `email` column and count occurrences.
- **Grouping and Counting**: By grouping the entries based on the `email` column and counting the number of occurrences, we can identify which emails appear more than once.
- **HAVING Clause**: We use the `HAVING` clause after `GROUP BY` to filter out emails that occur only once, thus listing only those that are duplicates.

### 中文:

目標是在 `Person` 表中識別出重複的電子郵件。考慮到每個電子郵件欄位均非空，並且表中包含主鍵（`id`），我們可以有效地按 `email` 列對記錄進行分組並計算出現次數。
- **分組與計算次數**：通過基於 `email` 列對條目進行分組並計算出現次數，我們可以識別哪些電子郵件出現了不止一次。
- **HAVING 子句**：我們在 `GROUP BY` 之後使用 `HAVING` 子句過濾掉只出現一次的電子郵件，從而只列出那些重複的。

## SQL Code

```sql
SELECT email AS Email
FROM Person
GROUP BY email
HAVING COUNT(*) > 1;
