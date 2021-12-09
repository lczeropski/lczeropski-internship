--Employees Earning More Than Their Managers
SELECT a.name as Employee FROM Employee as a , Employee as b
WHERE a.ManagerId = b.Id AND a.Salary > b.Salary;