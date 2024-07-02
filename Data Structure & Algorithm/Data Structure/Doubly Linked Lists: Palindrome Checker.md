# Doubly Linked Lists: Palindrome Checker

## Description and Goal

Write a method to determine whether a given doubly linked list reads the same forwards and backwards.

For example, if the list contains the values [1, 2, 3, 2, 1], then the method should return True, since the list is a palindrome.

If the list contains the values [1, 2, 3, 4, 5], then the method should return False, since the list is not a palindrome.

建立一個函數來確定給定的雙向鏈表是否正反讀都是一樣的。

例如，如果鏈表包含值 [1, 2, 3, 2, 1]，返回 True，因為鏈表是回文。

如果鏈表包含值 [1, 2, 3, 4, 5]，返回 False，因為鏈表不是回文。

## Method Name

### is_palindrome

## Problem Solution Approach

### English:

To check if a doubly linked list is a palindrome:

1. **Initialize Pointers**: Set `left` to the head and `right` to the tail of the list.
2. **Compare Values**: Compare the values of the nodes at `left` and `right` pointers.
3. **Move Pointers**: Move `left` to the next node and `right` to the previous node.
4. **Loop Until Middle**: Continue until the `left` and `right` pointers meet or cross.
5. **Check for Equality**: If all corresponding values are equal, the list is a palindrome.

### 中文:

檢查雙向鏈表是否為回文：

1. **初始化指針**：設置 `left` 為頭節點，`right` 為尾節點。
2. **比較值**：比較 `left` 和 `right` 指針所指節點的值。
3. **移動指針**：將 `left` 移動到下一個節點，`right` 移動到上一個節點。
4. **迴圈直到中間**：繼續此過程直到 `left` 和 `right` 指針相遇或交錯。
5. **檢查相等性**：如果所有對應的值都相等，則鏈表是回文。

## Code Implementation
* **Time Complexity = O(n)**

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self, value):
        new_node = Node(value)
        self.head = new_node
        self.tail = new_node
        self.length = 1

    def print_list(self):
        temp = self.head
        while temp is not None:
            print(temp.value, end=' ')
            temp = temp.next
        print()
        
    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.length += 1
        return True
    
    def is_palindrome(self):
        left = self.head
        right = self.tail
        while left is not None and right is not None and left != right and left.prev != right:
            if left.value != right.value:
                return False
            left = left.next
            right = right.prev
        return True
```
