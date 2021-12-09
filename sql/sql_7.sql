--Duplicate Emails
SELECT email FROM Person
GROUP BY email
HAVING COUNT(Email) > 1;