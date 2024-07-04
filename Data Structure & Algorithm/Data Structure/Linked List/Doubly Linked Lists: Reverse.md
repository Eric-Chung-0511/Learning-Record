# Doubly Linked Lists: Reverse

## Description and Goal

Create a method called `reverse` that reverses the order of the nodes in the doubly linked list, i.e., the first node becomes the last node, the second node becomes the second-to-last node, and so on. The values of the nodes are not changed, only their pointers are reversed.

建立一個反轉的雙向鍊錶讓原本的第一個變成最後一個，第二個變倒數第二個，以此類推。

## Method Name

### reverse

## Problem Solution Approach

### English:

To reverse a doubly linked list:

1. **Initialize Pointer**: Set a pointer `temp` to the head of the list.
2. **Traverse and Swap**: Traverse the list and swap the `prev` and `next` pointers of each node.
3. **Move Pointer**: Move the pointer to the new previous node (which was the next node before the swap).
4. **Update Head and Tail**: Once the traversal is complete, swap the head and tail pointers.

### 中文:

反轉雙向鏈表：

1. **初始化指針**：設置指針 `temp` 為鏈表的頭部。
2. **遍歷和交換**：遍歷鏈表，交換每個節點的 `prev` 和 `next` 指針。
3. **移動指針**：將指針移動到新的前一個節點（即交換前的下一個節點）。
4. **更新頭和尾**：遍歷完成後，交換頭指針和尾指針。

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
    
    def reverse(self):
        temp = self.head
        while temp is not None:
            temp.prev, temp.next = temp.next, temp.prev
            temp = temp.prev
        self.head, self.tail = self.tail, self.head
```
