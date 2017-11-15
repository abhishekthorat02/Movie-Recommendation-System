from datetime import datetime
import numpy as np
import pandas as pd
import math
import sys

actors = pd.read_csv("task1/database/phase1_dataset/imdb-actor-info.csv")
tags = pd.read_csv("task1/database/phase1_dataset/mltags.csv")
movies = pd.read_csv("task1/database/phase1_dataset/movie-actor.csv")

actor_movie = pd.merge(actors, movies, left_on='id', right_on='actorid', how='inner')
actor_movie = actor_movie.drop(['name', 'gender', 'actorid', 'actor_movie_rank'], axis=1)

actor_tags = pd.merge(actor_movie, tags, on='movieid')
actor_tags = actor_tags.drop(['userid', 'timestamp'], axis=1)
tagid = tags.drop(['userid', 'movieid', 'timestamp'], axis=1)
tagid = tagid.groupby('tagid', as_index=False).agg(lambda x: set(x))

actor_tags1 = actor_tags
actor_tags = actor_tags.groupby('id',as_index = False).agg(lambda x: list(x))

tags_map = {}
for i in range(0, len(tagid)):
    tags_map[tagid.loc[i][0]] = i
actors = actors.drop(['name', 'gender'], axis=1)
actors_map = {}
for i in range(0, len(actors)):
    actors_map[actors.loc[i][0]] = i
tag_set = set("")
for row in range(0, len(actor_tags)):
    tag_set = tag_set.union(actor_tags.iloc[row]['tagid'])

tag_set = list(tag_set)
actor_list = list(set(actors['id']))
timestamp = list(set(tags['timestamp']))
for i in range(0, len(timestamp)):
    timestamp[i] = (datetime.strptime(timestamp[i], "%Y-%m-%d %H:%M:%S") - datetime.utcfromtimestamp(0)).total_seconds()

df = pd.DataFrame(timestamp)
norm = (df - df.min()) / (df.max() - df.min())

act_tag = np.zeros(shape=(len(actors), len(tags)), dtype=float)

# Normalize the timestamp of the tags
timestamp_norm = pd.DataFrame(columns=['tagid', 'timestamp'])
timestamp_norm['tagid'] = list(set(tags['tagid']))
timestamp_norm['timestamp'] = norm
timestamp_norm.set_index('tagid', inplace=True)

# Calculate the time-weighted tf-idf value of actors and tags
for id in range(0, len(actor_tags)):
    tag_list = actor_tags.loc[id][2]
    for tag in tag_list:
        tfcount = float(tag_list.count(tag)) / len(tag_list)
        count = 0
        for id1 in range(0, len(actor_tags)):
            if tag in actor_tags.loc[id1][2]:
                count += 1
        idfcount = math.log10(float(len(actor_tags)) / count)
        tfidf = tfcount * idfcount
        act_tag[actors_map[actor_tags.loc[id][0]]][tags_map[tag]] = tfidf + timestamp_norm.loc[tag][0]

# Create the actor-actor similarity matrix
D = np.array(act_tag)
D_T = np.transpose(D)
act_act = np.dot(D, D_T)

# Get the input seed actors and create teleportation matrix
seeds = []
for i in range(1, len(sys.argv)):
    seeds.append(int(sys.argv[i]))
teleport = np.zeros(shape=(len(actors), 1), dtype=float)
for i in range(0, len(seeds)):
    teleport[actors_map[seeds[i]]] = 1.0 / len(seeds)
    i += 1

alpha = 0.85
err = 0.001
m = 100
# Normalize the similarity matrix along columns to 1 to use as transition matrix
act_act_sum = act_act.sum(axis=0)
for i in range(len(act_act)):
    for j in range(len(act_act)):
        if act_act_sum[i] != 0:
            act_act[i][j] /= act_act_sum[i]
act_act_norm = act_act

size = act_act.shape[0]
t = np.array(teleport)
pagerank = np.ones(size)
prev = np.zeros(size)

# Calculate the pagerank
while np.sum(np.abs(pagerank - prev)) > err:
    prev = pagerank
    pagerank = (alpha * np.dot(act_act_norm, pagerank)) + ((1 - alpha) * t)

actor_pagerank = pd.DataFrame(columns=['actor','pagerank'])
actor_pagerank['actor'] = actors
actor_pagerank['pagerank'] = pagerank
actor_pagerank = actor_pagerank.sort_values(by='pagerank', ascending=False)

top_actors = actor_pagerank[(-actor_pagerank.actor.isin(seeds))]
top_actors.set_index('actor', inplace=True)
print top_actors.head(n=10).reset_index()
