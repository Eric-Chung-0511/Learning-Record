## Description and Goal

Write a solution to report the first name, last name, city, and state of each person in the Person table. If the address of a personId is not present in the Address table, report null instead.

## Tables

**Table: Person**

| Column Name | Type    |
|-------------|---------|
| personId    | int     |
| lastName    | varchar |
| firstName   | varchar |

**Table: Address**

| Column Name | Type    |
|-------------|---------|
| addressId   | int     |
| personId    | int     |
| city        | varchar |
| state       | varchar |

## Example Input and Output

**Input:**

Person table:
| personId | lastName | firstName |
|----------|----------|-----------|
| 1        | Wang     | Allen     |
| 2        | Alice    | Bob       |

Address table:
| addressId | personId | city          | state      |
|-----------|----------|---------------|------------|
| 1         | 2        | New York City | New York   |
| 2         | 3        | Leetcode      | California |

**Output:**

| firstName | lastName | city          | state    |
|-----------|----------|---------------|----------|
| Allen     | Wang     | Null          | Null     |
| Bob       | Alice    | New York City | New York |

## Thoughts and Solution Approach

## English
We need to retrieve each person's first name and last name from the Person table, and also their city and state information from the Address table. 
If a personId in the Person table does not have corresponding address information in the Address table, the city and state should be returned as NULL.

Query Method:
LEFT JOIN: This type of join ensures that all records from the Person table are selected. 
If there is a matching personId in the Address table, the associated city and state are selected; if no match exists, then the city and state are displayed as NULL.

## 中文
我們需要從 'Person'表中獲取每個人的姓氏與姓名，同時在'Address'表中獲取城市與州的訊息。如果'Person'表中某個'personId'在'Address'表中沒有對應訊息，則對應城市與州返回'NULL'。

查詢方法:
LEFT JOIN: 這能保證'Person'表中所有訊息記錄皆被選出。如果'Address'表中存在匹配的'personId'相關的城市與州會被選出;如果沒有則返回'NULL'。
因為'LEFT JOIN'會返回所有行並與右表匹配，如右表沒有匹配，右表相關數值則返回'NULL'。

## SQL Code

```sql
SELECT
    p.firstName,
    p.lastName,
    a.city,
    a.state
FROM
    Person AS p
LEFT JOIN
    Address AS a ON p.personId = a.personId;

