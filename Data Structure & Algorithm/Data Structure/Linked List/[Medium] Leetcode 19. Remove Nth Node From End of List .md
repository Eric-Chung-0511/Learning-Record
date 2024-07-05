# 19. Remove Nth Node From End of List 

## Problem Description 
**English**:
Given the head of a linked list, remove the nth node from the end of the list and return its head.

**中文**:
給定鏈表的頭結點，移除倒數第 n 個節點並返回其頭結點。

## Solution Explanation 

**English**:
To solve this problem, we can use the two-pointer technique. We use a dummy node to simplify edge cases where the head might be removed. We first move the `first` pointer n+1 steps ahead, then move both `first` and `second` pointers until `first` reaches the end. Finally, we skip the desired node by updating the `next` pointer of the `second` pointer.

**中文**:
為了解決這個問題，我們可以使用雙指針技術。我們使用一個虛擬節點來簡化可能移除頭結點的邊界情況。首先將 `first` 指針前移 n+1 步，然後同時移動 `first` 和 `second` 指針直到 `first` 到達末端。最後，我們通過更新 `second` 指針的 `next` 指針來跳過所需的節點。

## Detailed Steps 

**English**:
1. **Create a Dummy Node**: Create a dummy node `dummy` and set its `next` to the head of the list.
2. **Initialize Pointers**: Use pointers `first` and `second` both pointing to `dummy`.
3. **Move First Pointer**: Move the `first` pointer n+1 steps ahead.
4. **Move Both Pointers**: Move both `first` and `second` pointers until `first` reaches the end of the list.
5. **Skip the Node**: Set `second.next` to `second.next.next` to remove the desired node.
6. **Return the Head**: Return `dummy.next` which is the new head of the list.

**中文**:
1. **創建虛擬節點**：創建一個虛擬節點 `dummy`，並將其 `next` 設置為鏈表的頭結點。
2. **初始化指針**：使用指針 `first` 和 `second` 都指向 `dummy`。
3. **移動第一個指針**：將 `first` 指針前移 n+1 步。
4. **同時移動兩個指針**：同時移動 `first` 和 `second` 指針直到 `first` 到達鏈表的末端。
5. **跳過節點**：設置 `second.next` 為 `second.next.next` 以移除所需的節點，也就是忽略該節點。
6. **返回頭結點**：返回 `dummy.next`，這是鏈表的新頭結點。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the linked list. This is because we traverse the list twice, once to move the `first` pointer and once to find the node to remove.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是鏈表中的節點數。這是因為我們遍歷鏈表兩次，一次是移動 `first` 指針，另一次是找到要移除的節點。

## Visual Explanation 

### English:

Suppose we have a linked list:

- `head`: 1 -> 2 -> 3 -> 4 -> 5, and `n = 2`

#### Initial State
- `dummy` points to a new node with value 0, `dummy.next` points to `head`.
- `first` and `second` both point to `dummy`.

#### Move First Pointer
- Move `first` pointer 3 steps ahead (n+1):
    - Step 1: `first` points to 1.
    - Step 2: `first` points to 2.
    - Step 3: `first` points to 3.

#### Move Both Pointers
- Move `first` and `second` until `first` reaches the end:
    - Move 1: `first` points to 4, `second` points to 1.
    - Move 2: `first` points to 5, `second` points to 2.
    - Move 3: `first` points to None, `second` points to 3.

#### Skip the Node
- `second.next = second.next.next` removes the node with value 4.

Final list: 1 -> 2 -> 3 -> 5

### 中文：

假設我們有一個鏈表：

- `head`: 1 -> 2 -> 3 -> 4 -> 5, 且 `n = 2`

#### 初始狀態
- `dummy` 指向一個值為 0 的新節點，`dummy.next` 指向 `head`。
- `first` 和 `second` 都指向 `dummy`。

#### 移動第一個指針
- 將 `first` 指針前移 3 步（n+1）：
    - 第一步：`first` 指向 1。
    - 第二步：`first` 指向 2。
    - 第三步：`first` 指向 3。

#### 同時移動兩個指針
- 同時移動 `first` 和 `second` 指針直到 `first` 到達末端：
    - 第一步：`first` 指向 4，`second` 指向 1。
    - 第二步：`first` 指向 5，`second` 指向 2。
    - 第三步：`first` 指向 None，`second` 指向 3。

#### 跳過節點
- `second.next = second.next.next` 移除值為 4 的節點。

最終鏈表：1 -> 2 -> 3 -> 5

## Code Implementation 

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def removeNthFromEnd(self, head, n):
    dummy = ListNode(0)
    dummy.next = head
    first = dummy
    second = dummy

    for _ in range(n + 1):
        first = first.next
    while first is not None:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next

