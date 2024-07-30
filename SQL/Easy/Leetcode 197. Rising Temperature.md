# SQL Challenge: Leetcode 197. Rising Temperature

## Description and Goal

Write a solution to find all dates' IDs where the temperature was higher than on the previous day.

## Tables

**Table: Weather**

| Column Name  | Type |
|--------------|------|
| id           | int  |
| recordDate   | date |
| temperature  | int  |

**id** is a column with unique values for this table. Each row represents the temperature on a specific day.

## Example Input and Output

### Example 1:

**Input:**

Weather table:
| id  | recordDate  | temperature |
|-----|-------------|-------------|
| 1   | 2015-01-01  | 10          |
| 2   | 2015-01-02  | 25          |
| 3   | 2015-01-03  | 20          |
| 4   | 2015-01-04  | 30          |

**Output:**

| id |
|----|
| 2  |
| 4  |

**Explanation:**
- On 2015-01-02, the temperature was higher than on the previous day (from 10°C on 2015-01-01 to 25°C).
- On 2015-01-04, the temperature increased from the previous day (from 20°C on 2015-01-03 to 30°C).

## Thoughts and Solution Approach

### English:

The task requires comparing the temperature of each day with that of the previous day.
- **Self-Join**: Use a self-join on the `Weather` table to compare the records of consecutive days.
- **`DATEDIFF` Usage**: `DATEDIFF` is a SQL function that calculates the difference in days between two dates. In this query, it's used to ensure that the records being compared are exactly one day apart.
- **Temperature Comparison**: After identifying consecutive days, check if the temperature of the current day (`a.recordDate`) is greater than the previous day (`b.recordDate`).

### 中文:

題目要求比較每一天與前一天的溫度。
- **與自己連接**：在 `Weather` 與自己做連接，比較連續兩天的記錄。
- **`DATEDIFF` 使用**：`DATEDIFF`用來計算兩個日期之間的天數差異。在此查詢中，它用來確保被比較的記錄正好相隔一天。
- **溫度比較**：在確定了連續的兩天後，檢查當前天（`a.recordDate`）的溫度是否高於前一天（`b.recordDate`）。

## SQL Code

```sql
SELECT a.id
FROM Weather a, Weather b
WHERE DATEDIFF(a.recordDate, b.recordDate) = 1
AND a.temperature > b.temperature;
