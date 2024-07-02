# Doubly Linked Lists: Swap First and Last 

## Description and Goal

Swap the values of the first and last node in a doubly linked list. The pointers to the nodes themselves are not swapped - only their values are exchanged.

將雙向鏈表中第一個節點和最後一個節點的值交換。節點本身的指針不會被交換，僅交換它們的值。

## Problem Solution Approach

### English:

To swap the values of the first and last node in a doubly linked list:

1. **Check for Empty List**: If the list is empty (`self.head` is `None`), return immediately.
2. **Check for Single Node**: If the list contains only one node (`self.head == self.tail`), no swap is needed.
3. **Swap Values**: Exchange the values of the `head` node and the `tail` node.

### 中文:

要在雙向鏈表中交換第一個節點和最後一個節點的值：

1. **檢查空列表**：如果鏈表為空 (`self.head` 為 `None`)，直接返回。
2. **檢查單節點**：如果鏈表中只有一個節點 (`self.head == self.tail`)，則不需要交換。
3. **交換值**：交換 `head` 節點和 `tail` 節點的值。

## Code Implementation

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
            print(temp.value)
            temp = temp.next
        
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
    
    def swap_first_last(self):
        # If the list is empty or has only one node, no need to swap
        if self.head is None or self.head == self.tail:
            return
        
        # Swap the values of the head and tail nodes
        self.head.value, self.tail.value = self.tail.value, self.head.value
```
