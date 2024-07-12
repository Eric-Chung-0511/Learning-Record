# 530. Minimum Absolute Difference in BST 

## Problem Description 

**English**:
Given the root of a binary search tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.

**中文**:
給定二叉搜索樹（BST）的根節點，返回樹中任意兩個不同節點值之間的最小絕對差。

## Solution Explanation 

**English**:
To solve this problem, we can perform an in-order traversal of the BST to get the node values in sorted order. Once we have the sorted list of node values, we can find the minimum absolute difference between any two consecutive values.

**中文**:
為了解決這個問題，我們可以對 BST 進行中序遍歷以獲取節點值的有序列表。一旦我們有了節點值的有序列表，就可以找到任意兩個連續值之間的最小絕對差。

## Detailed Steps 

**English**:
1. **Initialize List**: Create an empty list `inorder_list` to store the in-order traversal values.(left node --> root --> right node)
2. **In-order Traversal**:
    - Perform an in-order traversal of the tree and append the node values to `inorder_list`.
3. **Calculate Minimum Difference**:
    - Initialize `min_diff` to infinity.
    - Iterate through the `inorder_list` and find the minimum absolute difference between consecutive values.
4. **Return the Result**: Return the value of `min_diff`.

**中文**:
1. **初始化列表**：創建一個空列表 `inorder_list` 來存儲中序遍歷的值。（左子樹－> 根節點 －> 右子樹)
2. **中序遍歷**：
    - 先遞歸遍歷左子樹，再到根節點將值加入`inorder_list`，最後再遞歸遍歷右節點。
3. **計算最小差異**：
    - 將 `min_diff` 初始化為無限大，目的是確保比較過程中任何差值都會比初始值小。
    - 遍歷 `inorder_list`，找到連續值之間的最小絕對差。
4. **返回結果**：返回 `min_diff` 的值。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the tree. This is because we perform an in-order traversal and then iterate through the list of node values.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是樹中的節點數量。這是因為我們進行中序遍歷，然後遍歷節點值列表。

## Code Implementation 

```python
class Solution:
    def getMinimumDifference(self, root):
        inorder_list = []

        def inorder_traversal(node):
            if not node:
                return
            inorder_traversal(node.left)
            inorder_list.append(node.val)
            inorder_traversal(node.right)

        inorder_traversal(root)
        min_diff = float('inf')
        
        for i in range(1, len(inorder_list)):
            min_diff = min(min_diff, inorder_list[i] - inorder_list[i-1])
        
        return min_diff
