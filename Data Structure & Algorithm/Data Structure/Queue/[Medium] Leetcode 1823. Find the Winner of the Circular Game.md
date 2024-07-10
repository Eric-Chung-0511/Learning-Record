# 1823. Find the Winner of the Circular Game 

## Problem Description 

**English**:
Given the number of friends `n` and an integer `k`, find the winner of the game. In the game, friends are arranged in a circle and counted out every `k`-th friend is removed from the circle until only one friend remains. The winner is the last remaining friend.

**中文**:
給定朋友的數量 `n` 和一個整數 `k`，找出遊戲的贏家。在遊戲中，朋友們按圓圈排列，每數到第 `k` 個朋友時將其從圓圈中移除，直到只剩下一個朋友。贏家是最後剩下的朋友。

## Solution Explanation 

**English**:
To solve this problem, we use a list to simulate the circle of friends. We iterate through the list, removing the `k`-th friend until only one friend remains. The position of the friend to be removed is determined by the formula `(index + k - 1) % len(friends)`.

**中文**:
為了解決這個問題，我們使用一個列表來模擬朋友的圓圈。我們迭代該列表，每次移除第 `k` 個朋友，直到只剩下一個朋友。要移除的朋友的位置由公式 `(index + k - 1) % len(friends)` 確定。

## Detailed Steps 

**English**:
1. **Initialize List**: Create a list `friends` containing integers from 1 to `n`.
2. **Initialize Index**: Set the starting index to 0.
3. **Iterate and Remove**:
    - While the length of `friends` is greater than 1:
        - Calculate the index of the friend to be removed using `(index + k - 1) % len(friends)`.
        - Remove the friend at the calculated index.
4. **Return the Winner**: The last remaining element in the list is the winner.

**中文**:
1. **初始化列表**：創建一個包含從 1 到 `n` 的整數的列表 `friends`。
2. **初始化索引**：將起始索引設置為 0。
3. **迭代並移除**：
    - 當 `friends` 的長度大於 1 時：
        - 使用 `(index + k - 1) % len(friends)` 計算要移除的朋友的索引。
        - 移除計算出的索引處的朋友。
4. **返回贏家**：列表中最後剩下的元素即為贏家。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n \* k)** in the worst case, where `n` is the number of friends and `k` is the step count. This is because we iterate through the list and remove elements repeatedly.

**中文**:
該解決方案的時間複雜度在最壞情況下是 **O(n \* k)**，其中 `n` 是朋友的數量，`k` 是步長。這是因為我們反覆迭代列表並移除元素。

## Visual Explanation 

### English:

Suppose `n = 5` and `k = 2`:

1. **Initial List**: `[1, 2, 3, 4, 5]`
2. **First Iteration**: Remove `2`
   - List: `[1, 3, 4, 5]`
3. **Second Iteration**: Remove `4`
   - List: `[1, 3, 5]`
4. **Third Iteration**: Remove `1`
   - List: `[3, 5]`
5. **Fourth Iteration**: Remove `5`
   - List: `[3]`
6. **Winner**: `3`

### 中文：

假設 `n = 5` 和 `k = 2`：

1. **初始列表**：`[1, 2, 3, 4, 5]`
2. **第一次迭代**：移除 `2`
   - 列表：`[1, 3, 4, 5]`
3. **第二次迭代**：移除 `4`
   - 列表：`[1, 3, 5]`
4. **第三次迭代**：移除 `1`
   - 列表：`[3, 5]`
5. **第四次迭代**：移除 `5`
   - 列表：`[3]`
6. **贏家**：`3`

## Code Implementation 

```python
class Solution:
    def findTheWinner(self, n, k):
        friends = list(range(1, n+1))
        index = 0

        while len(friends) > 1:
            index = (index + k - 1) % len(friends)
            friends.pop(index)
        return friends[0]

# Example usage:
# n = 5, k = 2
# The winner of the game is 3

solution = Solution()
print(solution.findTheWinner(5, 2))  # Output: 3
