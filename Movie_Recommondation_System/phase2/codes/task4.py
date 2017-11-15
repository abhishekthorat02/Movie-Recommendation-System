import numpy as np
import pandas as pd
import sys

user = int(sys.argv[1])
tags = pd.read_csv("task1/database/phase1_dataset/mltags.csv")
movies_tags = tags.drop(['tagid', 'timestamp'], axis=1)
movies_tags = movies_tags.groupby('userid', as_index=False).agg(lambda x: list(x))
movies_tags.set_index('userid', inplace=True)

# seed movies
seeds = movies_tags.loc[user][0]

movies = pd.read_csv("task1/database/phase1_dataset/mlmovies.csv")
movies = movies.drop(['moviename', 'year', 'genres'], axis=1)
movies_tags = pd.merge(movies, tags, on = 'movieid')
movies_tags = movies_tags.drop(['userid','timestamp'], axis = 1)

movies_tags1 = movies_tags
movies_tags = movies_tags.groupby('movieid',as_index = False).agg(lambda x: set(x))

tag_set = set("")
for row in range(0, len(movies_tags)):
    tag_set = tag_set.union(movies_tags.iloc[row]['tagid'])
tag_list = list(tag_set)
movie_list = list(set(movies['movieid']))

df = pd.DataFrame(index = movie_list, columns = tag_list)
df = df.fillna(0)
for i in range(0,len(movies_tags1)):
    df.loc[movies_tags1.iloc[i]['movieid']][movies_tags1.iloc[i]['tagid']] = df.loc[movies_tags1.iloc[i]['movieid']][movies_tags1.iloc[i]['tagid']] + 1

# Create actor-tag matrix
D = np.array(df)
D_T = np.transpose(D)

# Create the movie-movie similarity matrix
movie_movie = np.dot(D, D_T)
movie_movie = movie_movie.astype(dtype=float)

# Create teleportation matrix with the seeds input
movies_map = {}
teleport = np.zeros(shape=(len(movies), 1), dtype=float)
for i in range(0, len(movies)):
    movies_map[movies.loc[i][0]] = i
for i in range(0, len(seeds)):
    teleport[movies_map[seeds[i]]] = 1.0 / len(seeds)
    i += 1

alpha = 0.85
err = 0.001
m = 100

# Normalize the similarity matrix along columns to 1 to use as transition matrix
movie_movie_sum = movie_movie.sum(axis=0)
for i in range(len(movie_movie)):
    for j in range(len(movie_movie)):
        if movie_movie_sum[i] != 0:
            movie_movie[i][j] /= movie_movie_sum[i]
movie_movie_norm = movie_movie

size = movie_movie.shape[0]
t = np.array(teleport)
pagerank = np.ones(size)
prev = np.zeros(size)

# Calculate the pagerank
while np.sum(np.abs(pagerank - prev)) > err:
    prev = pagerank
    pagerank = (alpha * np.dot(movie_movie_norm, pagerank)) + ((1 - alpha) * t)

movies_pagerank = pd.DataFrame(columns=['movies','pagerank'])
movies_pagerank['movies'] = movies
movies_pagerank['pagerank'] = pagerank
movies_pagerank = movies_pagerank.sort_values(by='pagerank', ascending=False)

top_movies = movies_pagerank[(-movies_pagerank.movies.isin(seeds))]
top_movies.set_index('movies', inplace=True)
print top_movies.head(n=5).reset_index()
