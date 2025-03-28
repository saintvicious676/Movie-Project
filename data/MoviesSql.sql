CREATE TABLE links (
    movieId INT PRIMARY KEY,
    imdbId INT,
    tmdbId INT
);

CREATE TABLE movies (
    movieId INT PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT NOT NULL
);

CREATE TABLE ratings (
    userId INT,
    movieId INT,
    rating DECIMAL(2,1) CHECK (rating BETWEEN 0 AND 5),
    timestamp BIGINT,
    PRIMARY KEY (userId, movieId, timestamp),
    FOREIGN KEY (movieId) REFERENCES movies(movieId) ON DELETE CASCADE
);

CREATE TABLE tags (
    userId INT,
    movieId INT,
    tag TEXT NOT NULL,
    timestamp BIGINT,
    PRIMARY KEY (userId, movieId, timestamp),
    FOREIGN KEY (movieId) REFERENCES movies(movieId) ON DELETE CASCADE
);

CREATE INDEX idx_ratings_movieId ON ratings (movieId);


CREATE INDEX idx_ratings_userId ON ratings (userId);


CREATE INDEX idx_ratings_timestamp ON ratings (timestamp);


CREATE INDEX idx_tags_movieId ON tags (movieId);


CREATE INDEX idx_tags_userId ON tags (userId);


CREATE INDEX idx_ratings_user_movie ON ratings (userId, movieId);

CREATE INDEX idx_tags_user_movie ON tags (userId, movieId);

CREATE INDEX idx_tags_full_text ON tags USING gin(to_tsvector('english', tag));

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

REFRESH MATERIALIZED VIEW popular_movies;