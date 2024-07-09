# 654. Maximum Binary Tree ( ** Interview Question)

## Problem Description / 問題描述

**English**:
Given an integer array `nums`, construct a maximum binary tree as described below:
- The root is the maximum number in `nums`.
- The left subtree is the maximum tree constructed from the elements left of the maximum number.
- The right subtree is the maximum tree constructed from the elements right of the maximum number.

Return the root of the maximum binary tree.

**中文**:
給定一個整數數組 `nums`，構造一個最大二叉樹，具體如下：
- 根結點是 `nums` 中的最大數字。
- 左子樹是由最大數字左側的元素構造的最大二叉樹。
- 右子樹是由最大數字右側的元素構造的最大二叉樹。

返回最大二叉樹的根結點。

## Solution Explanation / 解決方案解釋

**English**:
To solve this problem, we use a stack to build the tree. We traverse the `nums` array and for each number, we create a new tree node. We maintain the stack to ensure that the largest number is always at the top of the stack. If the current number is larger than the number at the top of the stack, we pop from the stack and set the current node's left child to be the popped node. Then, if the stack is not empty, we set the right child of the node at the top of the stack to be the current node. Finally, we push the current node onto the stack.

**中文**:
為了解決這個問題，我們使用堆疊來構建樹。我們遍歷 `nums` 數組，對於每個數字，我們創建一個新的樹節點。我們維護堆疊以確保最大的數字總是位於堆疊的頂部。如果當前數字大於堆疊頂部的數字，我們從堆疊中彈出，並將當前節點的左子節點設置為彈出的節點。然後，如果堆疊不為空，我們將堆疊頂部節點的右子節點設置為當前節點。最後，我們將當前節點推入堆疊。

## Detailed Steps / 詳細步驟

**English**:
1. **Initialize Stack**: Create an empty stack.
2. **Traverse the Array**:
    - For each number in `nums`, create a new tree node.
    - While the stack is not empty and the value of the node at the top of the stack is less than the current number, pop from the stack and set the current node's left child to be the popped node.
    - If the stack is not empty, set the right child of the node at the top of the stack to be the current node.
    - Push the current node onto the stack.
3. **Return the Root**: The first element in the stack is the root of the maximum binary tree.

**中文**:
1. **初始化堆疊**：創建一個空堆疊。
2. **遍歷數組**：
    - 對於 `nums` 中的每個數字，創建一個新的樹節點。
    - 當堆疊不為空且堆疊頂部節點的值小於當前數字時，從堆疊中彈出，並將當前節點的左子節點設置為彈出的節點。
    - 如果堆疊不為空，將堆疊頂部節點的右子節點設置為當前節點。
    - 將當前節點推入堆疊。
3. **返回根結點**：堆疊中的第一個元素是最大二叉樹的根結點。

## Time Complexity / 時間複雜度

**English**:
The time complexity of this solution is **O(n)**, where `n` is the length of the input array `nums`. This is because each element is pushed and popped from the stack at most once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是輸入數組 `nums` 的長度。這是因為每個元素最多被推入和彈出堆疊一次。

## Visual Explanation / 視覺化解釋

### English:

Suppose we have an array: `[3, 2, 1, 6, 0, 5]`

#### Initial State
- `stack` is empty.

#### First Iteration (num = 3)
- Create node with value 3.
- Push node onto stack.
- `stack`: [3]

#### Second Iteration (num = 2)
- Create node with value 2.
- Push node onto stack.
- `stack`: [3, 2]

#### Third Iteration (num = 1)
- Create node with value 1.
- Push node onto stack.
- `stack`: [3, 2, 1]

#### Fourth Iteration (num = 6)
- Create node with value 6.
- Pop 1 from stack, set 1 as left child of 6.
- Pop 2 from stack, set 2 as left child of 6.
- Pop 3 from stack, set 3 as left child of 6.
- Push node 6 onto stack.
- `stack`: [6]

#### Continue this process for remaining numbers.


### 中文：

假設我們有一個數組：`[3, 2, 1, 6, 0, 5]`

#### 初始狀態
- `stack` 是空的。

#### 第一次迭代 (num = 3)
- 創建值為 3 的節點。
- 將節點推入堆疊。
- `stack`: [3]

#### 第二次迭代 (num = 2)
- 創建值為 2 的節點。
- 將節點推入堆疊。
- `stack`: [3, 2]

#### 第三次迭代 (num = 1)
- 創建值為 1 的節點。
- 將節點推入堆疊。
- `stack`: [3, 2, 1]

#### 第四次迭代 (num = 6)
- 創建值為 6 的節點。
- 從堆疊中彈出 1，將 1 設為 6 的左子節點。
- 從堆疊中彈出 2，將 2 設為 6 的左子節點。
- 從堆疊中彈出 3，將 3 設為 6 的左子節點。
- 將節點 6 推入堆疊。
- `stack`: [6]

#### 繼續對剩餘數字進行此過程。



