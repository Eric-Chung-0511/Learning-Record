# Linked List: Remove Duplicates

## Description and Goal

Implement a method called `remove_duplicates()` within the `LinkedList` class that removes all duplicate values from a singly linked list containing integer values. The method should modify the existing list in-place and preserve the relative order of the nodes.

定義一個名為 `remove_duplicates()` 的函數，用於刪除單鏈表中所有重複的整數值。該方法應直接修改現有的列表，並保留節點的相對順序。

## Problem Solution Approach

### English:

To remove duplicates from a linked list using a set:

1. **Initialize a Set**: Use a set to keep track of seen values.
2. **Two Pointers**: Use two pointers, `current` to traverse the list and `prev` to point to the previous node.
3. **Traversal and Removal**: Traverse the list, and for each node, check if its value is in the set.
   - If the value is in the set, adjust `prev.next` to skip the current node.
   - If the value is not in the set, add it to the set and move the `prev` pointer.
4. **Continue Traversal**: Move the `current` pointer to the next node and repeat until the end of the list.

### 中文:

使用 Set 從鏈表中刪除重複值：

1. **初始化一個 Set**：使用一個 Set 來追蹤已經見過的值。
2. **兩個指針**：使用兩個指針，`current` 用於遍歷列表，`prev` 指向前一個節點。
3. **遍歷和移除**：遍歷列表，對於每個節點，檢查其值是否在 Set 中。
   - 如果值在 Set 中，調整 `prev.next` 跳過當前節點。
   - 如果值不在 Set 中，將其添加到 Set 中，並移動 `prev` 指針。
4. **繼續遍歷**：移動 `current` 指針到下一個節點，重複直到列表末尾。

## Code Implementation

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def remove_duplicates(self):
        if not self.head:
            return
        
        current = self.head
        prev = None
        seen = set()
        
        while current:
            if current.value in seen:
                prev.next = current.next
            else:
                seen.add(current.value)
                prev = current
            current = current.next
