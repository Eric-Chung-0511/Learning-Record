# 232. Implement Queue using Stacks 

## Problem Description 

**English**:
Implement a first-in-first-out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (`push`, `pop`, `peek`, and `empty`).

**中文**:
僅使用兩個堆棧實現一個先入先出（FIFO）的隊列。實現的隊列應支持正常隊列的所有功能（`push`、`pop`、`peek` 和 `empty`）。

## Solution Explanation 

**English**:
To solve this problem, we use two stacks:
1. `s1` is used to store the queue elements in the correct order.
2. `s2` is used as a temporary stack to reverse the order of elements.

When pushing an element, we first move all elements from `s1` to `s2`, then push the new element onto `s1`, and finally move all elements back from `s2` to `s1`. This way, the oldest element always remains at the bottom of `s1`.

**中文**:
為了解決這個問題，我們使用兩個堆棧：
1. `s1` 用來按正確的順序存儲隊列元素。
2. `s2` 作為臨時堆棧來反轉元素順序。

當推入一個元素時，我們首先將所有元素從 `s1` 移到 `s2`，然後將新元素推入 `s1`，最後將所有元素從 `s2` 移回 `s1`。這樣，最舊的元素總是位於 `s1` 的底部。

## Detailed Steps 

**English**:
1. **Initialize Stacks**: Create two empty stacks, `s1` and `s2`.
2. **Push Operation**:
    - Move all elements from `s1` to `s2`.
    - Push the new element onto `s1`.
    - Move all elements back from `s2` to `s1`.
3. **Pop Operation**: Remove and return the top element from `s1`.
4. **Peek Operation**: Return the top element of `s1` without removing it.
5. **Empty Operation**: Check if `s1` is empty.

**中文**:
1. **初始化堆棧**：創建兩個空堆棧，`s1` 和 `s2`。
2. **Push 操作**：
    - 將所有元素從 `s1` 移到 `s2`。
    - 將新元素推入 `s1`。
    - 將所有元素從 `s2` 移回 `s1`。
3. **Pop 操作**：移除並返回 `s1` 的頂部元素。
4. **Peek 操作**：返回 `s1` 的頂部元素但不移除它。
5. **Empty 操作**：檢查 `s1` 是否為空。

## Time Complexity 

**English**:
The time complexity of the `push` operation is **O(n)**, where `n` is the number of elements in the queue, because we move elements between the stacks. The `pop`, `peek`, and `empty` operations have a time complexity of **O(1)**.

**中文**:
`push` 操作的時間複雜度是 **O(n)**，其中 `n` 是隊列中的元素數量，因為我們在堆棧之間移動元素。`pop`、`peek` 和 `empty` 操作的時間複雜度是 **O(1)**。

## Visual Explanation 

### English:

Suppose we perform the following operations:

- `push(1)`
- `push(2)`
- `peek()`
- `pop()`
- `empty()`

#### Initial State
- `s1`: []
- `s2`: []

#### Push 1
- Move elements from `s1` to `s2`.
- `s1`: []
- `s2`: []
- Push 1 onto `s1`.
- `s1`: [1]
- Move elements back from `s2` to `s1`.
- `s1`: [1]
- `s2`: []

#### Push 2
- Move elements from `s1` to `s2`.
- `s1`: []
- `s2`: [1]
- Push 2 onto `s1`.
- `s1`: [2]
- Move elements back from `s2` to `s1`.
- `s1`: [2, 1]
- `s2`: []

#### Peek
- Return the top element of `s1`: 1

#### Pop
- Remove and return the top element of `s1`: 1
- `s1`: [2]

#### Empty
- Check if `s1` is empty: False

### 中文：

假設我們執行以下操作：

- `push(1)`
- `push(2)`
- `peek()`
- `pop()`
- `empty()`

#### 初始狀態
- `s1`: []
- `s2`: []

#### Push 1
- 將元素從 `s1` 移到 `s2`。
- `s1`: []
- `s2`: []
- 將 1 推入 `s1`。
- `s1`: [1]
- 將元素從 `s2` 移回 `s1`。
- `s1`: [1]
- `s2`: []

#### Push 2
- 將元素從 `s1` 移到 `s2`。
- `s1`: []
- `s2`: [1]
- 將 2 推入 `s1`。
- `s1`: [2]
- 將元素從 `s2` 移回 `s1`。
- `s1`: [2, 1]
- `s2`: []

#### Peek
- 返回 `s1` 的頂部元素：1

#### Pop
- 移除並返回 `s1` 的頂部元素：1
- `s1`: [2]

#### Empty
- 檢查 `s1` 是否為空：否

## Code Implementation 

```python
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []
        
    def push(self, x):
        while self.s1:
            self.s2.append(self.s1.pop())
        self.s1.append(x)
        while self.s2:
            self.s1.append(self.s2.pop())

    def pop(self):
        return self.s1.pop()
        
    def peek(self):
        return self.s1[-1]
        
    def empty(self):
        return not self.s1

# Example usage:
# obj = MyQueue()
# obj.push(1)
# obj.push(2)
# print(obj.peek())  # Output: 1
# print(obj.pop())   # Output: 1
# print(obj.empty()) # Output: False

queue = MyQueue()
queue.push(1)
queue.push(2)
print(queue.peek())  # Output: 1
print(queue.pop())   # Output: 1
print(queue.empty()) # Output: False
