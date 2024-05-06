# SQL Challenge: Leetcode 620. Not Boring Movies

## Description and Goal

* Write a solution to identify movies from a database where each movie has an odd-numbered ID and a description that does not include the term "boring". 
* The results should be ordered by the rating of the movies in descending order, showcasing the best-rated movies first.

## Tables

**Table: Cinema**

| Column Name   | Type    |
|---------------|---------|
| id            | int     |
| movie         | varchar |
| description   | varchar |
| rating        | float   |

- **id**: Primary key, an integer that uniquely identifies a movie.
- **movie**: Title of the movie.
- **description**: A brief description of the movie.
- **rating**: Movie's rating on a scale from 0 to 10, precise up to two decimal places.

## Example Input and Output

### Example 1:

**Input:**

Cinema table:
| id  | movie      | description | rating |
|-----|------------|-------------|--------|
| 1   | War        | great 3D    | 8.9    |
| 2   | Science    | fiction     | 8.5    |
| 3   | irish      | boring      | 6.2    |
| 4   | Ice song   | Fantacy     | 8.6    |
| 5   | House card | Interesting | 9.1    |

**Output:**

| id | movie      | description | rating |
|----|------------|-------------|--------|
| 5  | House card | Interesting | 9.1    |
| 1  | War        | great 3D    | 8.9    |

**Explanation:**
Movies with IDs 1, 3, and 5 are odd-numbered. Out of these, only IDs 1 and 5 have descriptions that are not "boring".

## Thoughts and Solution Approach

### English:

The challenge focuses on filtering the `Cinema` table to find non-boring movies with odd-numbered IDs:
- **Modulo Operator**: Use `id % 2 = 1` to filter for movies with odd-numbered IDs.
- **Description Check**: Include `description != 'boring'` to exclude any movie described as boring.
- **Ordering**: Sort the results by the `rating` column in descending order to show the highest-rated movies at the top.

### 中文:

這個題目著重於過濾 `Cinema` 表來找出描述不是 "boring" 且具有奇數 ID 的電影：
- **模數運算符**：使用 `id % 2 = 1` 篩選具有奇數 ID 的電影。
- **描述檢查**：包括 `description != 'boring'` 以排除任何描述為 "boring" 的電影。
- **排序**：按 `rating` 欄位降序排序，使最高評分的電影顯示在最前面。

## SQL Code

```sql
SELECT * FROM Cinema
WHERE id % 2 = 1 AND description != 'boring'
ORDER BY rating DESC;
