# 933. Number of Recent Calls 

## Problem Description 

**English**:
Write a class `RecentCounter` that counts recent requests within a certain time frame. Implement the `RecentCounter` class with the following methods:
- `__init__()`: Initializes the counter with an empty request queue.
- `ping(t)`: Adds a new request at time `t` and returns the number of requests that have happened in the past 3000 milliseconds (including the new request).

**中文**:
編寫一個計數最近請求的類 `RecentCounter`。實現 `RecentCounter` 類，該類具有以下方法：
- `__init__()`: 用空的請求隊列初始化計數器。
- `ping(t)`: 在時間 `t` 添加一個新請求，並返回過去 3000 毫秒內發生的請求數（包括新請求）。

## Solution Explanation 

**English**:
To solve this problem, we use a deque to store the timestamps of requests. When a new request comes in, we add its timestamp to the deque and remove any timestamps that are more than 3000 milliseconds older than the current timestamp. Finally, we return the size of the deque, which represents the number of recent requests.

**中文**:
為了解決這個問題，我們使用雙端隊列（deque）來存儲請求的時間戳。當有新請求到達時，我們將其時間戳添加到 deque 中，並移除任何超過當前時間戳 3000 毫秒的時間戳。最後，我們返回 deque 的大小，即表示最近請求的數量。

## Detailed Steps / 詳細步驟

**English**:
1. **Initialize Queue**: Create an empty deque to store the timestamps of the requests.
2. **Ping Method**:
    - Append the current timestamp `t` to the deque.
    - Remove timestamps from the front of the deque that are older than `t - 3000` milliseconds.
    - Return the size of the deque, which represents the number of recent requests.

**中文**:
1. **初始化隊列**：創建一個空的 deque 來存儲請求的時間戳。
2. **Ping 方法**：
    - 將當前時間戳 `t` 添加到 deque 中。
    - 移除 deque 前端超過 `t - 3000` 毫秒的時間戳。
    - 返回 deque 的大小，即表示最近請求的數量。

## Time Complexity / 時間複雜度

**English**:
The time complexity of the `ping` method is **O(1)** for appending and removing elements. In the worst case, all elements in the deque are within the 3000 milliseconds window, so we only iterate through the deque once for each `ping` call.

**中文**:
`ping` 方法的時間複雜度是 **O(1)**，因為添加和移除元素都是常數時間操作。在最壞情況下，deque 中的所有元素都在 3000 毫秒的窗口內，因此我們每次 `ping` 調用只會遍歷 deque 一次。

## Visual Explanation 

### English:

Suppose we perform the following pings:

- `ping(1)`
- `ping(100)`
- `ping(3001)`
- `ping(3002)`

#### Initial State
- `requests`: []

#### Ping 1
- Append 1 to `requests`.
- `requests`: [1]
- Number of recent calls: 1

#### Ping 100
- Append 100 to `requests`.
- Do not remove any number, No number is < -2900 (100-3000).
- `requests`: [1, 100]
- Number of recent calls: 2

#### Ping 3001
- Append 3001 to `requests`.
-  Do not remove any number,  No number is < 1 (3001-3000).
- `requests`: [1, 100, 3001]
- Number of recent calls: 3

#### Ping 3002
- Append 3002 to `requests`.
- Remove 1 from `requests` , since 1 < 2 (3002 - 3000).
- `requests`: [100, 3001, 3002]
- Number of recent calls: 3

### 中文：

假設我們執行以下 ping 操作：

- `ping(1)`
- `ping(100)`
- `ping(3001)`
- `ping(3002)`

#### 初始狀態
- `requests`: []

#### Ping 1
- 將 1 添加到 `requests`。
- `requests`: [1]
- 最近請求數量：1

#### Ping 100
- 將 100 添加到 `requests`。
- 不用移除任何數，沒有數 < -2900 (100 - 3000)。
- `requests`: [1, 100]
- 最近請求數量：2

#### Ping 3001
- 將 3001 添加到 `requests`。
- 不用移除任何數， 沒有數 < 1 (3001 - 3000)。
- `requests`: [1, 100, 3001]。
- 最近請求數量：3

#### Ping 3002
- 將 3002 添加到 `requests`。
- 移除 1，因為 1 < 2 (3002 - 3000）。
- `requests`: [100, 3001, 3002]
- 最近請求數量：3

## Code Implementation 

```python
from collections import deque

class RecentCounter:
    def __init__(self):
        self.requests = deque()
        
    def ping(self, t):
        self.requests.append(t)
        while self.requests[0] < t - 3000:
            self.requests.popleft()
        return len(self.requests)

# Example usage:
# counter = RecentCounter()
# print(counter.ping(1))    # Output: 1
# print(counter.ping(100))  # Output: 2
# print(counter.ping(3001)) # Output: 3
# print(counter.ping(3002)) # Output: 3
