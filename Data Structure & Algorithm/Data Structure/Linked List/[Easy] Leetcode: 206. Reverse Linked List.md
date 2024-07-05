# 206. Reverse Linked List 

## Problem Description 

**English**:
Given the head of a singly linked list, reverse the list, and return the reversed list.

**中文**:
給定單鏈表的頭結點，反轉鏈表並返回反轉後的鏈表。

## Solution Explanation 

**English**:
To solve this problem, we need to traverse the linked list and reverse the direction of the pointers for each node. We can do this by using three pointers: `prev`, `current` (head), and `temp` to store the next node. As we traverse the list, we reverse the `next` pointer of each node.

**中文**:
為了解決這個問題，我們需要遍歷鏈表並反轉每個節點的指針方向。我們可以使用三個指針：`prev`、`current`（head）和 `temp` 來存儲下一個節點。當我們遍歷鏈表時，反轉每個節點的 `next` 指針。

## Detailed Steps 

**English**:
1. **Initialize Pointers**: Set `prev` to `None` and `current` to `head`.
2. **Traverse the List**:
    - Store the next node in `temp`.
    - Reverse the `next` pointer of the `current` node to point to `prev`.
    - Move `prev` to `current`.
    - Move `current` to `temp`.
3. **Return the New Head**: Once the traversal is complete, `prev` will be the new head of the reversed list.

**中文**:
1. **初始化指針**：設置 `prev` 為 `None`，`current` 為 `head`。
2. **遍歷鏈表**：
    - 將下一個節點存儲在 `temp` 中。
    - 將 `current` 節點的 `next` 指針反轉指向 `prev`。
    - 將 `prev` 移動到 `current`。
    - 將 `current` 移動到 `temp`。
3. **返回新頭結點**：遍歷完成後，`prev` 將成為反轉後的鏈表的新頭結點。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the linked list. This is because we traverse each node exactly once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是鏈表中的節點數。這是因為我們每個節點只遍歷一次。

## Visual Explanation / 視覺化解釋

### English:

Suppose we have a linked list:

- `head`: 1 -> 2 -> 3 -> 4 -> 5

#### Initial State
- `prev` is `None`.
- `current` is `head` (1).

#### First Iteration
- `temp` is 2.
- Reverse `current.next` to `prev` (1 -> `None`).
- Move `prev` to `current` (1).
- Move `current` to `temp` (2).

#### Second Iteration
- `temp` is 3.
- Reverse `current.next` to `prev` (2 -> 1).
- Move `prev` to `current` (2).
- Move `current` to `temp` (3).

Continue this process until `current` is `None`.

Final list: 5 -> 4 -> 3 -> 2 -> 1

### 中文：

假設我們有一個鏈表：

- `head`: 1 -> 2 -> 3 -> 4 -> 5

#### 初始狀態
- `prev` 是 `None`。
- `current` 是 `head`（1）。

#### 第一次迭代
- `temp` 是 2。
- 將 `current.next` 反轉指向 `prev`（1 -> `None`）。
- 將 `prev` 移動到 `current`（1）。
- 將 `current` 移動到 `temp`（2）。

#### 第二次迭代
- `temp` 是 3。
- 將 `current.next` 反轉指向 `prev`（2 -> 1）。
- 將 `prev` 移動到 `current`（2）。
- 將 `current` 移動到 `temp`（3）。

繼續此過程直到 `current` 是 `None`。

最終鏈表：5 -> 4 -> 3 -> 2 -> 1

## Code Implementation 

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(self, head):
    prev = None

    while head:
        temp = head.next

        # reverse the pointer
        head.next = prev

        prev = head
        head = temp
    return prev
