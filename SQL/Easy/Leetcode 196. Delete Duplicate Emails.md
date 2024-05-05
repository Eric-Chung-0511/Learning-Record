# SQL Challenge: Leetcode 196. Delete Duplicate Emails

## Description and Goal

Write a solution to delete all duplicate emails from the `Person` table, keeping only the entry with the smallest `id` for each unique email.

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
| id  | email            |
|-----|------------------|
| 1   | john@example.com |
| 2   | bob@example.com  |
| 3   | john@example.com |

**Output:**

Person table after running the delete operation:
| id  | email            |
|-----|------------------|
| 1   | john@example.com |
| 2   | bob@example.com  |

**Explanation:**
`john@example.com` appears twice. The record with the smallest `id` (id = 1) is kept, while the other duplicate (`id` = 3) is deleted.

## Thoughts and Solution Approach

### English:

The task requires a `DELETE` statement that will remove duplicate emails from the `Person` table, ensuring only the entry with the smallest `id` remains for each email.
- **Using a Self-Join**: The solution involves a self-join on the `Person` table to identify duplicates.
- **Condition for Deletion**: The deletion condition `p1.email = p2.email and p1.id > p2.id` ensures that for any two records with the same email, the one with the greater `id` is deleted, preserving only the record with the smallest `id`.

### 中文:

題目要求寫一個 `DELETE` 語句，從 `Person` 表中刪除重複的電子郵件，確保每個電子郵件只保留具有最小 `id` 的郵件。
- **使用自我連接**：解決方案包括在 `Person` 表上進行自連接以識別重複項。
- **刪除條件**：刪除條件 `p1.email = p2.email and p1.id > p2.id` 確保對於任何兩條具有相同電子郵件的記錄，有更大 `id` 的那條會被刪除，只保留有最小 `id` 的記錄。

## SQL Code

```sql
DELETE p1 FROM Person p1, Person p2
WHERE p1.email = p2.email AND p1.id > p2.id;
