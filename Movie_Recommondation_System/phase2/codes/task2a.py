import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
import math
#import networkx

actors = pd.read_csv("task1/database/phase1_dataset/imdb-actor-info.csv")  # load data
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

act_tag = [[0 for y in range(len(tags))] for x in range(len(actors))]
act_tag = np.zeros(shape=(len(actors), len(tags)), dtype=float)

timestamp_norm = pd.DataFrame(columns=['tagid', 'timestamp'])
timestamp_norm['tagid'] = list(set(tags['tagid']))
timestamp_norm['timestamp'] = norm
timestamp_norm.set_index('tagid', inplace=True)

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
#print(np.unique(act_tag))
D = np.array(act_tag)
D_T = np.transpose(D)
act_act = np.dot(D, D_T)    # actor-actor similarity matrix
#print(act_act)

U, s, V = linalg.svd(act_act)   # SVD implemenetation 
''' above implementation giving a sparse U matrix with most values = 0 '''
svd = TruncatedSVD(n_components=3)
U1 = svd.fit_transform(act_act)
#print(s)
#inverse = svd.inverse_transform(U1)
#print(inverse)
#print (U1[:,0].shape)
latent_semantics = pd.DataFrame({'actorid':actor_list})
latent_semantics['latent_semantic_1'] = U1[:,0]
latent_semantics['latent_semantic_2'] = U1[:,1]
latent_semantics['latent_semantic_3'] = U1[:,2]


print("Latent Semantic 1")
print(latent_semantics[['actorid','latent_semantic_1']].sort_values(['latent_semantic_1'], ascending = False))

print("Latent Semantic 2")
print(latent_semantics[['actorid','latent_semantic_2']].sort_values(['latent_semantic_2'], ascending = False))

print("Latent Semantic 3")
print(latent_semantics[['actorid','latent_semantic_3']].sort_values(['latent_semantic_3'], ascending = False))

latent_semantics = latent_semantics.drop(['actorid'], axis = 1)
samples = latent_semantics.as_matrix()
#print(samples)
kmeans = KMeans(n_clusters = 3)
cluster = kmeans.fit_predict(samples)
#print(cluster)

grouping_df = pd.DataFrame({'actorid':actor_list})
grouping_df['cluster_id'] = cluster

grouping_df = grouping_df.groupby('cluster_id',as_index = False).agg(lambda x: list(x))
#print(grouping_df)

for i in range(0,3):
    print("Group " +str( i+1))
    print(grouping_df.iloc[i]['actorid'])
