-- Combine 2 tables
SELECT lastName, firstName, city, state FROM Person 
LEFT JOIN Address ON Person.personId = Address.personId;
--
{"headers":{"Person":["personId","lastName","firstName"],"Address":["addressId","personId","city","state"]},"rows":{"Person":[[1,"Wang","Allen"],[2,"Alice","Bob"]],"Address":[[1,2,"New York City","New York"],[2,3,"Leetcode","California"]]}}