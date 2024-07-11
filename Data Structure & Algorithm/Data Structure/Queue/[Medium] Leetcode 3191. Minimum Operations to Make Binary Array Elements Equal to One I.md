# 3191. Minimum Operations to Make Binary Array Elements Equal to One I 

## Problem Description 
**English**:
Given a binary array `nums`, you can flip three consecutive elements from 0 to 1 or from 1 to 0 in one operation. Return the minimum number of operations needed to make all elements in the array equal to 1. If it is not possible, return -1.

**中文**:
給定一個二進制數組 `nums`，你可以在一次操作中將三個連續元素從 0 翻轉為 1 或從 1 翻轉為 0。返回使數組中的所有元素都變為 1 所需的最小操作數。如果不可能，則返回 -1。

## Solution Explanation 

**English**:
To solve this problem, we iterate through the array and perform the flip operation whenever we encounter a 0. We flip the current element and the next two elements and count the operation. After processing the array, we check if all elements are 1. If yes, we return the count of operations; otherwise, we return -1.

**中文**:
為了解決這個問題，我們遍歷數組，當遇到 0 時執行翻轉操作。我們翻轉當前元素和接下來的兩個元素，並計算操作次數。處理完數組後，檢查所有元素是否為 1。如果是，我們返回操作次數；否則返回 -1。

## Detailed Steps 

**English**:
1. **Initialize Variables**: Create a variable `operation` to count the number of operations.
2. **Iterate through the Array**:
    - For each element from 0 to `n-3`:
        - In order to constraint the range, so use n-2.
        - If the current element is 0, flip it and the next two elements.
        - Increment the `operation` count.
3. **Check Result**:
    - If all elements are 1, return the `operation` count.
    - Otherwise, return -1.

**中文**:
1. **初始化變量**：創建一個變量 `operation` 來計算操作次數。
2. **遍歷數組**：
    - 對於從 0 到 `n-3` 的每個元素：
        - 為了不讓循環超過範圍，所以要(n-2)。
        - 如果當前元素為 0，翻轉它和接下來的兩個元素。
        - 增加 `operation` 計數。
3. **檢查結果**：
    - 如果所有元素都為 1，返回 `operation` 計數。
    - 否則返回 -1。

## Time Complexity / 時間複雜度

**English**:
The time complexity of this solution is **O(n)**, where `n` is the length of the array. This is because we iterate through the array once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是數組的長度。這是因為我們遍歷數組一次。

## Visual Explanation 

### English:

Suppose `nums = [0, 0, 1, 0, 1, 1, 0]`:

1. **Initial State**: `nums = [0, 0, 1, 0, 1, 1, 0]`
2. **First Iteration** (i = 0):
    - Flip elements at indices 0, 1, 2.
    - `nums`: `[1, 1, 0, 0, 1, 1, 0]`
    - `operation`: 1
3. **Second Iteration** (i = 2):
    - Flip elements at indices 2, 3, 4.
    - `nums`: `[1, 1, 1, 1, 0, 1, 0]`
    - `operation`: 2
4. **Third Iteration** (i = 4):
    - Flip elements at indices 4, 5, 6.
    - `nums`: `[1, 1, 1, 1, 1, 0, 1]`
    - `operation`: 3
5. **Check Result**: `nums` is not all 1s. Return -1.

### 中文：

假設 `nums = [0, 0, 1, 0, 1, 1, 0]`：

1. **初始狀態**：`nums = [0, 0, 1, 0, 1, 1, 0]`
2. **第一次迭代**（i = 0）：
    - 翻轉索引 0, 1, 2 處的元素。
    - `nums`：`[1, 1, 0, 0, 1, 1, 0]`
    - `operation`：1
3. **第二次迭代**（i = 2）：
    - 翻轉索引 2, 3, 4 處的元素。
    - `nums`：`[1, 1, 1, 1, 0, 1, 0]`
    - `operation`：2
4. **第三次迭代**（i = 4）：
    - 翻轉索引 4, 5, 6 處的元素。
    - `nums`：`[1, 1, 1, 1, 1, 0, 1]`
    - `operation`：3
5. **檢查結果**：`nums` 不是全 1。返回 -1。

## Code Implementation 

```python
class Solution:
    def minOperations(self, nums):
        n = len(nums)
        operation = 0
        for i in range(n-2):
            if nums[i] == 0:
                nums[i] = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                operation += 1

        if all(x == 1 for x in nums):
            return operation
        else:
            return -1

# Example usage:
# nums = [0, 0, 1, 0, 1, 1, 0]
# The minimum operations to make all elements 1 is -1 because it's impossible in this case.

solution = Solution()
print(solution.minOperations([0, 0, 1, 0, 1, 1, 0]))  # Output: -1
