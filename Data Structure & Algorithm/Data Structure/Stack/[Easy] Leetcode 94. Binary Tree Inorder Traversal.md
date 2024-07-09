# 94. Binary Tree Inorder Traversal 
## Problem Description 

**English**:
Given the root of a binary tree, return the inorder traversal of its nodes' values.

**中文**:
給定二叉樹的根結點，返回其節點值的中序遍歷。

## Solution Explanation 

**English**:
To solve this problem, we use an iterative approach with a stack to perform the inorder traversal of the binary tree. The inorder traversal visits nodes in the following order: left subtree, root, right subtree.

**中文**:
為了解決這個問題，我們使用帶堆疊的迭代方法來進行二叉樹的中序遍歷。中序遍歷按以下順序訪問節點：左子樹、根節點、右子樹。

## Detailed Steps

**English**:
1. **Initialize Data Structures**: Create an empty list `res` to store the result and a stack `stack` to help with traversal.
2. **Traverse the Tree**:
    - **Push Current Node to Stack**: Save the current node so that we can backtrack to it later.
    - **Traverse Left Subtree**: Move to the left child of the current node until there are no more left children.
    - **Pop Node from Stack**: When `curr` is `None`, it means we've reached the leftmost node and need to backtrack.
    - **Visit Node and Record Its Value**: Add the value of the popped node to the result list.
    - **Traverse Right Subtree**: Move to the right child of the current node and repeat the process.
3. **Return the Result**: After the traversal is complete, return the result list `res`.

**中文**:
1. **初始化數據結構**：創建一個空列表 `res` 來存儲結果，創建一個堆疊 `stack` 來幫助遍歷。
2. **遍歷樹**：
    - **將當前節點壓入堆疊**：保存當前節點，以便稍後可以回退到該節點。
    - **遍歷左子樹**：移動到當前節點的左子節點，直到沒有左子節點為止。
    - **從堆疊中彈出節點**：當 `curr` 為空時，說明已經到達最左端，需要回退到上一個節點。
    - **訪問節點並記錄其值**：將彈出的節點值加入結果列表。
    - **遍歷右子樹**：移動到當前節點的右子節點，並重複上述過程。
3. **返回結果**：遍歷完成後，返回結果列表 `res`。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the binary tree. This is because we visit each node exactly once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是二叉樹中的節點數。這是因為我們每個節點只訪問一次。

##  Steps Explanation 

### English：

#### Initial State
- `res` is empty.
- `stack` is empty.
- `curr` points to the root (1).

#### First Iteration
- Push 1 to stack.
- Move `curr` to 2.

#### Second Iteration
- Push 2 to stack.
- Move `curr` to 4.

#### Third Iteration
- Push 4 to stack.
- Move `curr` to `None`.

#### Fourth Iteration
- Pop 4 from stack.
- Add 4 to `res`.
- Move `curr` to `None`.

#### Fifth Iteration
- Pop 2 from stack.
- Add 2 to `res`.
- Move `curr` to 5.

Continue this process until all nodes are visited.

Final result: [4, 2, 5, 1, 3]

### 中文：

#### 初始狀態
- `res` 是空的。
- `stack` 是空的。
- `curr` 指向根結點（1）。

#### 第一次迭代
- 將 1 壓入堆疊。
- 將 `curr` 移動到 2。

#### 第二次迭代
- 將 2 壓入堆疊。
- 將 `curr` 移動到 4。

#### 第三次迭代
- 將 4 壓入堆疊。
- 將 `curr` 移動到 `None`。

#### 第四次迭代
- 從堆疊中彈出 4。
- 將 4 加入 `res`。
- 將 `curr` 移動到 `None`。

#### 第五次迭代
- 從堆疊中彈出 2。
- 將 2 加入 `res`。
- 將 `curr` 移動到 5。

繼續這個過程直到所有節點被訪問。

最終結果： [4, 2, 5, 1, 3]

## Code Implementation

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorderTraversal(self, root):
    res = []
    stack = deque()
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res

# Example usage:
# Tree:     1
#         /   \
#        2     3
#       / \
#      4   5
# Inorder Traversal: [4, 2, 5, 1, 3]

def build_tree():
    return TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))

tree_root = build_tree()
solution = Solution()
result = solution.inorderTraversal(tree_root)

print(result)  # Output: [4, 2, 5, 1, 3]
