# Leetcode 83. Remove Duplicates from Sorted List 

## Problem Description

**English**:
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

**中文**:
給定一個排序鏈表的頭結點，刪除所有重複的節點，使得每個元素只出現一次。返回排序後的鏈表。

## Solution Explanation 

**English**:
To solve this problem, we need to traverse the sorted linked list and remove any duplicates we encounter. This can be done in-place without using additional space for another list.

**中文**:
為了解決這個問題，需要遍歷已排序的鏈表並移除遇到的所有重複節點。這可以在原鍊表中完成，而不需要使用額外的空間來存儲另一個鏈表。

## Detailed Steps 

**English**
1. **Check if the List is Empty**: If the list is empty, return the head immediately.
2. **Initialize Pointer**: Use a pointer `current` to traverse the list starting from the head.
3. **Traverse the List**:
    - If the current node's value is the same as the next node's value, skip the next node by updating `current.next` to `current.next.next`.
    - If the current node's value is different from the next node's value, move the `current` pointer to the next node.
4. **Return the Head**: Once the traversal is complete, return the head of the modified list.

**中文**
1. **檢查鏈表是否為空**：如果鏈表為空，立即返回頭結點。
2. **初始化指針**：使用指針 `current` 從頭結點開始遍歷鏈表。
3. **遍歷鏈表**：
    - 如果當前節點的值與下一個節點的值相同，通過將 `current.next` 更新為 `current.next.next` 來跳過下一個節點。
    - 如果當前節點的值與下一個節點的值不同，將 `current` 指針移動到下一個節點。
4. **返回頭結點**：遍歷完成後，返回修改後的鏈表的頭結點。


## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the linked list. This is because we traverse each node exactly once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是鏈表中的節點數。這是因為我們每個節點只遍歷一次。

## Code Implementation 

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteDuplicates(head):
    if not head:
        return head
    
    current = head
    while current and current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    
    return head
