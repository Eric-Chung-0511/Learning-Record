# SQL Challenge: Leetcode 1661. Average Time of Process per Machine

## Description and Goal

Write a solution to calculate the average time each machine in a factory takes to complete a process. The time to complete a process is determined by subtracting the 'start' timestamp from the 'end' timestamp for each process. The average time is then calculated by dividing the total time to complete all processes on a machine by the number of processes run.

## Tables

**Table: Activity**

| Column Name    | Type    |
|----------------|---------|
| machine_id     | int     |
| process_id     | int     |
| activity_type  | enum    |
| timestamp      | float   |

- **machine_id**: ID of a machine.
- **process_id**: ID of a process running on the machine.
- **activity_type**: An ENUM of type ('start', 'end').
- **timestamp**: A float representing the current time in seconds.

## Example Input and Output

### Example 1:

**Input:**

Activity table:
| machine_id | process_id | activity_type | timestamp |
|------------|------------|---------------|-----------|
| 0          | 0          | start         | 0.712     |
| 0          | 0          | end           | 1.520     |
| 0          | 1          | start         | 3.140     |
| 0          | 1          | end           | 4.120     |
| 1          | 0          | start         | 0.550     |
| 1          | 0          | end           | 1.550     |
| 1          | 1          | start         | 0.430     |
| 1          | 1          | end           | 1.420     |
| 2          | 0          | start         | 4.100     |
| 2          | 0          | end           | 4.512     |
| 2          | 1          | start         | 2.500     |
| 2          | 1          | end           | 5.000     |

**Output:**

| machine_id | processing_time |
|------------|-----------------|
| 0          | 0.894           |
| 1          | 0.995           |
| 2          | 1.456           |

**Explanation:**
- Machine 0's average processing time is calculated as ((1.520 - 0.712) + (4.120 - 3.140)) / 2 = 0.894 seconds.
- Machine 1 and Machine 2 are calculated similarly.

## Thoughts and Solution Approach

### English:

To tackle this problem, the approach involves:
- **Self-Join on Activity Table**: A self-join is used on the `Activity` table to pair each 'start' record with its corresponding 'end' record for the same machine and process.
- **Time Calculation**: The difference between the 'end' and 'start' timestamps is calculated to determine the duration of each process.
- **Average Calculation**: The average processing time for each machine is then calculated using the `AVG()` function, and the result is rounded to three decimal places for clarity.

### 中文:

解決這個問題的方法包括：
- **在Activity表中JOIN**：使用`JOIN`在 `Activity` 表上，將每個 'start' 記錄與相同機器和過程的對應 'end' 記錄配對。
- **時間計算**：計算 'end' 和 'start' 時間戳之間的差異，以確定每個過程的持續時間。
- **平均時間計算**：使用 `AVG()` 函數計算每台機器的平均處理時間，並將結果四捨五入到三個小數位以便清晰顯示。

## SQL Code

```sql
SELECT a1.machine_id, ROUND(AVG(a2.timestamp - a1.timestamp), 3) AS processing_time
FROM Activity a1
JOIN Activity a2 ON a1.machine_id = a2.machine_id 
AND a1.process_id = a2.process_id
AND a1.activity_type = 'start'
AND a2.activity_type = 'end'
GROUP BY a1.machine_id;
