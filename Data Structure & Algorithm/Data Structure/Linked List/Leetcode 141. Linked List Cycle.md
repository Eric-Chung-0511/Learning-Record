# Leetcode 141. Linked List Cycle 

## Problem Description 

**English**:
Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Return `true` if there is a cycle in the linked list. Otherwise, return `false`.

**中文**:
給定鏈表的頭結點 `head`，判斷鏈表中是否存在循環。

如果在鏈表中存在某個節點，可以通過不斷地跟隨 `next` 指針再次到達該節點，則該鏈表中存在環。在內部，`pos` 用於表示尾部的 `next` 指針連接到的節點的索引。注意，`pos` 不會作為參數傳遞。

如果鏈表中存在環，則返回 `true`。否則，返回 `false`。

## Solution Explanation 

**English**:
To solve this problem, we use the Floyd’s Cycle-Finding Algorithm, also known as the "Tortoise and Hare" algorithm. This algorithm uses two pointers, `slow` and `fast`, which move at different speeds through the linked list. If there is a cycle, the `fast` pointer will eventually meet the `slow` pointer.

**中文**:
為了解決這個問題，我們使用弗洛伊德循環檢測算法，也稱為“龜兔賽跑”算法。此算法使用兩個指針，`slow` 和 `fast`，它們以不同的速度穿過鏈表。如果存在環，`fast` 指針最終將與 `slow` 指針相遇。

## Detailed Steps 

**English**:
1. **Initialize Pointers**: Set `fast` and `slow` pointers to the head of the list.
2. **Traverse the List**:
    - Using `fast` and `fast.next` is because if `fast` exists, `slow` must exist.
    - Move `slow` pointer one step at a time.
    - Move `fast` pointer two steps at a time.
    - If `slow` and `fast` pointers meet, a cycle is detected, return `true`.
3. **Return Result**: If `fast` pointer reaches the end of the list, return `false` as no cycle exists.

**中文**:
1. **初始化指針**：將 `fast` 和 `slow` 指針設置為鏈表的頭結點。
2. **遍歷鏈表**：
    - 用`fast`與`fast.next`是因為如果`fast`存在，`slow`一定存在。
    - 將 `slow` 指針每次移動一步。
    - 將 `fast` 指針每次移動兩步。
    - 如果 `slow` 和 `fast` 指針相遇，則檢測到環，返回 `true`。
3. **返回結果**：如果 `fast` 指針到達鏈表末端，返回 `false`，表示不存在循環。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the linked list. This is because each node is visited at most once by each pointer.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是鏈表中的節點數。這是因為每個節點最多被每個指針訪問一次。

## Code Implementation 

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def hasCycle(head):
    fast = head
    slow = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
