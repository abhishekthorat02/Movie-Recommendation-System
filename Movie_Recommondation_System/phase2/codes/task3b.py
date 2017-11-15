import pandas as pd
import numpy as np
import sys

actor_movies = pd.read_csv("task1/database/phase1_dataset/movie-actor.csv")
actors = pd.read_csv("task1/database/phase1_dataset/imdb-actor-info.csv")

actor_movies = actor_movies.drop(['actor_movie_rank'], axis=1)
actor_list = list(set(actors['id']))
df = pd.DataFrame(index=actor_list, columns=actor_list)
df = df.fillna(0)
co_co = np.array(df, dtype=float)
actor_movies = {k: g["movieid"].tolist() for k, g in actor_movies.groupby("actorid")}

# Calculate the coactor-coactor similarity matrix
for i in range(0, len(actor_movies)):
    for j in range(0, len(actor_movies)):
        actor1 = set(actor_movies[actor_list[i]])
        actor2 = set(actor_movies[actor_list[j]])
        count_set = actor1.intersection(actor2)
        co_co[i][j] = len(count_set)
        co_co[j][i] = len(count_set)

# Create a teleportaion with the seeds input
actors = actors.drop(['name', 'gender'], axis=1)
actors_map = {}

# Get the input seed actors
seeds = []
for i in range(1, len(sys.argv)):
    seeds.append(int(sys.argv[i]))
teleport = np.zeros(shape=(len(actors), 1), dtype=float)
for i in range(0, len(actors)):
    actors_map[actors.loc[i][0]] = i
for i in range(0, len(seeds)):
    teleport[actors_map[seeds[i]]] = 1.0 / len(seeds)
    i += 1

alpha = 0.85
err = 0.001
m = 100
# Normalize the similarity matrix along columns to 1 to use as transition matrix
co_co_norm = co_co / co_co.sum(axis=0)

size = co_co.shape[0]
t = np.array(teleport)
pagerank = np.ones(size)
prev = np.zeros(size)

# Calculate the pagerank
for i in range(m):
    prev = pagerank
    pagerank = ((1 - alpha) * np.dot(co_co_norm, pagerank)) + (alpha * t)

actor_pagerank = pd.DataFrame(columns=['actor','pagerank'])
actor_pagerank['actor'] = actors
actor_pagerank['pagerank'] = pagerank
actor_pagerank = actor_pagerank.sort_values(by='pagerank', ascending=False)

top_actors = actor_pagerank[(-actor_pagerank.actor.isin(seeds))]
top_actors.set_index('actor', inplace=True)
print top_actors.head(n=10).reset_index()
