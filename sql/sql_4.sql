-- Rank Scores
SELECT sc.Score,
       (SELECT COUNT(*)+1 FROM (SELECT DISTINCT Score FROM Scores)
        AS uniqeScores WHERE Score > sc.Score) AS 'rank' 
FROM Scores sc ORDER BY sc.Score DESC;