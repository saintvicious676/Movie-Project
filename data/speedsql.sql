--caching example
CREATE MATERIALIZED VIEW popular_movies AS
SELECT m.movieId, m.title, AVG(r.rating) AS avg_rating
FROM movies m
JOIN ratings r ON m.movieId = r.movieId
GROUP BY m.movieId
ORDER BY avg_rating DESC
LIMIT 10;

-- Create a materialized view that caches only ratings from the last 5 years
CREATE MATERIALIZED VIEW recent_ratings AS
SELECT r.userId, r.movieId, r.rating, r.timestamp
FROM ratings r
WHERE TO_TIMESTAMP(r.timestamp) >= NOW() - INTERVAL '5 years';

-- Create an index to speed up queries on the materialized view
CREATE INDEX idx_recent_ratings_movieId ON recent_ratings (movieId);
CREATE INDEX idx_recent_ratings_userId ON recent_ratings (userId);
-- Query the view
SELECT * FROM recent_ratings LIMIT 10;
-- Query the view
SELECT * FROM popular_movies;

--3 second query time

EXPLAIN ANALYZE 
SELECT m.movieId, m.title, AVG(r.rating) AS avg_rating
FROM movies m
JOIN ratings r ON m.movieId = r.movieId
WHERE TO_TIMESTAMP(r.timestamp) >= NOW() - INTERVAL '5 years'
GROUP BY m.movieId
ORDER BY avg_rating DESC
LIMIT 10;

--2.8 second query time

EXPLAIN ANALYZE 
SELECT * FROM popular_movies;
--.47 second query time

EXPLAIN ANALYZE 
SELECT * FROM recent_ratings WHERE movieId = 100;
--.42 second query time

