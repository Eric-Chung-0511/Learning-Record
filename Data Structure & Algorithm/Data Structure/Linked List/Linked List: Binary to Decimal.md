# Linked List: Binary to Decimal

## Description and Goal

Implement the `binary_to_decimal` method for the `LinkedList` class. This method should convert a binary number, represented as a linked list, to its decimal equivalent. Each node in the linked list contains a single digit (0 or 1) of the binary number, and the whole number is formed by traversing the linked list from the head to the end.

實現 `LinkedList` 類的 `binary_to_decimal` 方法。這個方法應該將用鏈表表示的二進制數轉換為十進制等價值。鏈表中的每個節點包含二進制數的一個數字（0 或 1），整個數字由從頭到尾遍歷鏈表形成。

## Method Name

### binary_to_decimal

## Problem Solution Approach

### English:

To convert a binary number represented as a linked list to its decimal equivalent:

1. **Initialize num**: Set `num` to 0 to store the resulting decimal number.
2. **Traverse the List**: Use a pointer `current` to traverse the linked list.
3. **Binary to Decimal Conversion**: For each node, multiply `num` by 2 and add `current.value` to `num`.
4. **Move Pointer**: Move `current` to the next node and repeat until the end of the list.
5. **Return Result**: When traversal is complete, `num` contains the decimal equivalent of the binary number.

### 中文:

將用鏈表表示的二進制數轉換為十進制數：

1. **初始化 num**：設置 `num` 為 0 來存儲最終的十進制數。
2. **遍歷鏈表**：使用指針 `current` 遍歷鏈表。
3. **二進制到十進制轉換**：對於每個節點，將 `num` 乘以 2 並將 `current.value` 加到 `num` 中。
4. **移動指針**：將 `current` 移動到下一個節點，重複直到列表末尾。
5. **返回結果**：當遍歷完成後，`num` 包含二進制數的十進制等價值。

## Code Implementation 
* **Time Complexity = O(n)**

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, value):
        if not self.head:
            self.head = ListNode(value)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = ListNode(value)

    def binary_to_decimal(self):
        num = 0
        current = self.head
        while current:
            num = num * 2 + current.value
            current = current.next
        return num

```

### Why Multiplying by 2 Converts Binary to Decimal

When converting a binary number to a decimal number, each digit in the binary number represents an increasing power of 2, starting from the rightmost digit (which represents \(2^0\)). By multiplying the current accumulated value by 2 at each step, you effectively shift the binary digits to the left, similar to how you would shift digits in decimal by multiplying by 10. This is equivalent to increasing the exponent in the power of 2 for each digit.

For example, in the binary number 101:
- The leftmost digit (1) represents \(2^2\)
- The middle digit (0) represents \(2^1\)
- The rightmost digit (1) represents \(2^0\)

When you traverse the linked list and multiply by 2, you're effectively adding each binary digit's contribution to the total decimal value by shifting its position:

1. Start with num = 0.
2. For the first digit (1): num = 0 * 2 + 1 = 1
3. For the second digit (0): num = 1 * 2 + 0 = 2
4. For the third digit (1): num = 2 * 2 + 1 = 5

This gives the final decimal value of 5 for the binary number 101.

### 為何二進制數乘以2能轉換成十進制

在將二進制數轉換為十進制數時，每個二進制數字位代表從右到左遞增的2的冪次方。通過在每一步將當前累積值乘以2，可將二進制數字向左移動，類似於在十進制數中乘以10來移動數字位置。這相當於每個數字增加其2的冪次方。

例如，在二進制數101中：
- 最左邊的數字（1）代表 \(2^2\)
- 中間的數字（0）代表 \(2^1\)
- 最右邊的數字（1）代表 \(2^0\)

當遍歷鏈表並將累積值乘以2時，實際上是在將每個二進制數字對應的十進制值加到總值中，通過移動位置來實現：

1. 初始化 num = 0。
2. 對於第一個數字（1）：num = 0 * 2 + 1 = 1
3. 對於第二個數字（0）：num = 1 * 2 + 0 = 2
4. 對於第三個數字（1）：num = 2 * 2 + 1 = 5

這樣，二進制數101的最終十進制值為5。
