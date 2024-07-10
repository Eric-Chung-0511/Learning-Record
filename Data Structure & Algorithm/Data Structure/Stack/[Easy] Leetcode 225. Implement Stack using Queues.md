# 225. Implement Stack using Queues 

## Problem Description 

**English**:
Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (`push`, `pop`, `top`, and `empty`).

**中文**:
僅使用兩個隊列實現一個後進先出（LIFO）的堆棧。實現的堆棧應支持正常堆棧的所有功能（`push`、`pop`、`top` 和 `empty`）。

## Solution Explanation 

**English**:
To implement a stack using two queues, we can use one queue (`queue1`) to store the elements in the correct order and another queue (`queue2`) to assist with the `push` operation. During the `push` operation, we transfer all elements from `queue1` to `queue2`, then swap the queues to maintain the order.

**中文**:
為了使用兩個隊列實現堆棧，我們可以使用一個隊列（`queue1`）來按正確順序存儲元素，另一個隊列（`queue2`）來協助 `push` 操作。在 `push` 操作期間，我們將所有元素從 `queue1` 轉移到 `queue2`，然後交換兩個隊列以保持順序。

## Detailed Steps 

**English**:
1. **Initialize Queues**: Create two empty queues `queue1` and `queue2`.
2. **Push Operation**:
    - Append the element to `queue2`.
    - Transfer all elements from `queue1` to `queue2`.
    - Swap `queue1` and `queue2`.
3. **Pop Operation**: Remove and return the front element from `queue1`, using `popleft`.
4. **Top Operation**: Return the front element from `queue1` without removing it.
5. **Empty Operation**: Check if `queue1` is empty.

**中文**:
1. **初始化隊列**：創建兩個空隊列 `queue1` 和 `queue2`。
2. **Push 操作**：
    - 將元素追加到 `queue2`。
    - 將所有元素從 `queue1` 轉移到 `queue2`。
    - 交換 `queue1` 和 `queue2`。
3. **Pop 操作**：從 `queue1` 中移除並返回首個元素，使用`popleft`。
4. **Top 操作**：返回 `queue1` 中的首個元素但不移除它。
5. **Empty 操作**：檢查 `queue1` 是否為空。

## Time Complexity 

**English**:
The time complexity of the `push` operation is **O(n)** due to the transfer of elements between the queues. The `pop`, `top`, and `empty` operations are **O(1)**.

**中文**:
由於在隊列之間轉移元素，`push` 操作的時間複雜度是 **O(n)**。`pop`、`top` 和 `empty` 操作的時間複雜度是 **O(1)**。

## Visual Explanation 

### English:

Suppose we perform the following operations:

- `push(1)`
- `push(2)`
- `top()`
- `pop()`
- `empty()`

#### Initial State
- `queue1`: []
- `queue2`: []

#### Push 1
- Append 1 to `queue2`: `queue2` = [1]
- Swap `queue1` and `queue2`: `queue1` = [1], `queue2` = []

#### Push 2
- Append 2 to `queue2`: `queue2` = [2]
- Transfer elements from `queue1` to `queue2`: `queue2` = [2, 1], `queue1` = []
- Swap `queue1` and `queue2`: `queue1` = [2, 1], `queue2` = []

#### Top
- Return the front element of `queue1`: 2

#### Pop
- Remove and return the front element of `queue1`: 2
- `queue1` = [1]

#### Empty
- Check if `queue1` is empty: False

### 中文：

假設我們執行以下操作：

- `push(1)`
- `push(2)`
- `top()`
- `pop()`
- `empty()`

#### 初始狀態
- `queue1`: []
- `queue2`: []

#### Push 1
- 將 1 追加到 `queue2`：`queue2` = [1]
- 交換 `queue1` 和 `queue2`：`queue1` = [1]，`queue2` = []

#### Push 2
- 將 2 追加到 `queue2`：`queue2` = [2]
- 將元素從 `queue1` 轉移到 `queue2`：`queue2` = [2, 1]，`queue1` = []
- 交換 `queue1` 和 `queue2`：`queue1` = [2, 1]，`queue2` = []

#### Top
- 返回 `queue1` 的首個元素：2

#### Pop
- 移除並返回 `queue1` 的首個元素：2
- `queue1` = [1]

#### Empty
- 檢查 `queue1` 是否為空：False

## Code Implementation 

```python

class MyStack:
    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x):
        self.queue2.append(x)
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1
        
    def pop(self):
        return self.queue1.popleft()

    def top(self):
        return self.queue1[0]

    def empty(self):
        return not self.queue1

# Example usage:
# stack = MyStack()
# stack.push(1)
# stack.push(2)
# print(stack.top())    # Output: 2
# print(stack.pop())    # Output: 2
# print(stack.empty())  # Output: False
