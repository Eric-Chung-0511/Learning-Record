# 155. Min Stack

## Problem Description 

**English**:
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time. Implement the `MinStack` class with the following methods:
- `push(val)`: Pushes the element `val` onto the stack.
- `pop()`: Removes the element on the top of the stack.
- `top()`: Gets the top element of the stack.
- `getMin()`: Retrieves the minimum element in the stack.

You must implement a solution with `O(1`) time complexity for each function.

**中文**:
設計一個支持 push、pop、top 和在常數時間內檢索最小元素的堆棧。實現 `MinStack` 類，該類具有以下方法：
- `push(val)`: 將元素 `val` 推入堆棧。
- `pop()`: 移除堆棧頂部的元素。
- `top()`: 獲取堆棧頂部的元素。
- `getMin()`: 檢索堆棧中的最小元素。
  
所有步驟必須在時間複雜度 `O(1)`以內完成。

## Solution Explanation 

**English**:
To solve this problem, we use two stacks:
1. A primary stack (`stack`) to store all elements.
2. A secondary stack (`min_stack`) to store the minimum values. This secondary stack is updated only when a new minimum value is pushed or when the current minimum value is popped from the primary stack.

**中文**:
為了解決這個問題，我們使用兩個堆棧：
1. 一個主堆棧（`stack`）用來存儲所有元素。
2. 一個輔助堆棧（`min_stack`）用來存儲最小值。只有在推入新的最小值或從主堆棧中彈出當前最小值時才更新此輔助堆棧。

## Detailed Steps 

**English**:
1. **Initialize Stacks**: Create two empty stacks, `stack` and `min_stack`.
2. **Push Operation**:
    - Append the value to `stack`.
    - If `min_stack` is empty or the value is less than or equal to the top element of `min_stack`, append the value to `min_stack`.
3. **Pop Operation**:
    - Remove and return the top element from `stack`.
    - If the removed element is equal to the top element of `min_stack`, remove the top element from `min_stack`.
4. **Top Operation**: Return the top element of `stack`.
5. **GetMin Operation**: Return the top element of `min_stack`.

**中文**:
1. **初始化堆棧**：創建兩個空堆棧，`stack` 和 `min_stack`。
2. **Push 操作**：
    - 將值追加到 `stack`。
    - 如果 `min_stack` 為空或該值小於或等於 `min_stack` 的頂部元素，則將該值追加到 `min_stack`。
3. **Pop 操作**：
    - 移除並返回 `stack` 的頂部元素。
    - 如果移除的元素等於 `min_stack` 的頂部元素，則移除 `min_stack` 的頂部元素。
4. **Top 操作**：返回 `stack` 的頂部元素。
5. **GetMin 操作**：返回 `min_stack` 的頂部元素。

## Time Complexity 

**English**:
All operations (`push`, `pop`, `top`, `getMin`) have a time complexity of **O(1)**.

**中文**:
所有操作（`push`、`pop`、`top`、`getMin`）的時間複雜度都是 **O(1)**。

## Visual Explanation 

### English:

Suppose we perform the following operations:

- `push(5)`
- `push(3)`
- `push(7)`
- `push(2)`
- `getMin()`
- `pop()`
- `getMin()`

#### Initial State
- `stack`: []
- `min_stack`: []

#### Push 5
- `stack`: [5]
- `min_stack`: [5]

#### Push 3
- `stack`: [5, 3]
- `min_stack`: [5, 3]

#### Push 7
- `stack`: [5, 3, 7]
- `min_stack`: [5, 3]

#### Push 2
- `stack`: [5, 3, 7, 2]
- `min_stack`: [5, 3, 2]

#### GetMin
- Return the top element of `min_stack`: 2

#### Pop
- Remove 2 from `stack`: [5, 3, 7]
- Remove 2 from `min_stack`: [5, 3]

#### GetMin
- Return the top element of `min_stack`: 3

### 中文：

假設我們執行以下操作：

- `push(5)`
- `push(3)`
- `push(7)`
- `push(2)`
- `getMin()`
- `pop()`
- `getMin()`

#### 初始狀態
- `stack`: []
- `min_stack`: []

#### Push 5
- `stack`: [5]
- `min_stack`: [5]

#### Push 3
- `stack`: [5, 3]
- `min_stack`: [5, 3]

#### Push 7
- `stack`: [5, 3, 7]
- `min_stack`: [5, 3]

#### Push 2
- `stack`: [5, 3, 7, 2]
- `min_stack`: [5, 3, 2]

#### GetMin
- 返回 `min_stack` 的頂部元素：2

#### Pop
- 從 `stack` 中移除 2： [5, 3, 7]
- 從 `min_stack` 中移除 2： [5, 3]

#### GetMin
- 返回 `min_stack` 的頂部元素：3

## Code Implementation 

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
        
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            top_element = self.stack.pop()
            if top_element == self.min_stack[-1]:
                self.min_stack.pop()

    def top(self):
        return self.stack[-1] if self.stack else None

    def getMin(self):
        return self.min_stack[-1] if self.min_stack else None
