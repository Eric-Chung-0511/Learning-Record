# SQL Challenge: Leetcode 178.Ranking Scores

## Description and Goal

Write a solution to find the rank of the scores in the Scores table. The ranking should be calculated such that:
- Scores are ranked from highest to lowest.
- Ties are given the same rank.
- The next ranking number after a tie should be the next consecutive integer (i.e., no gaps in ranking).
Return the result table ordered by score in descending order.

## Tables

**Table: Scores**

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| score       | decimal |

**id** is the primary key for this table. Each row contains the score of a game, which is a floating point value with two decimal places.

## Example Input and Output

### Example 1:

**Input:**

Scores table:
| id   | score |
|------|-------|
| 1    | 3.50  |
| 2    | 3.65  |
| 3    | 4.00  |
| 4    | 3.85  |
| 5    | 4.00  |
| 6    | 3.65  |

**Output:**

| score | rank |
|-------|------|
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |

## Thoughts and Solution Approach

### English:
* To solve this problem, we need to rank the scores in the table from highest to lowest.
  
* We can use SQL window functions like `DENSE_RANK()` to assign rankings. This function ensures that ties receive the same rank, and ranks increase without gaps after ties.

* In this practice, 'Rank' is keyword, so need to add quote on it.

### 中文
* 要解決這問題需要對分數進行排名，從最高至最低。

* 可以使用窗口函數的`DENSE_RANK()`來分配排名，使用這函數能確保相同分數的排名相同時不會產生間隙，會繼續逐一列出。

* 因為 'RANK' 是關鍵字，所以必須加引號。

```sql
SELECT score, DENSE_RANK() OVER (ORDER BY score DESC) AS 'RANK'
FROM Scores
ORDER BY score DESC;
