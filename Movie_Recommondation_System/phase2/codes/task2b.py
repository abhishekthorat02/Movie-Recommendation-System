import pandas as pd
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans

actor_movies = pd.read_csv("task1/database/phase1_dataset/movie-actor.csv") # load data
actors = pd.read_csv("task1/database/phase1_dataset/imdb-actor-info.csv")

actor_movies = actor_movies.drop(['actor_movie_rank'], axis = 1)
#actor_movies = actor_movies.groupby('actorid',as_index = False).agg(lambda x: set(x))
actor_list = list(set(actors['id']))
df = pd.DataFrame(index = actor_list, columns = actor_list)
df = df.fillna(0)
co_co = np.array(df)
actor_movies = {k: g["movieid"].tolist() for k,g in actor_movies.groupby("actorid")}  # for each actor find set of movies
#print(actor_movies)

for i in range(0,len(actor_movies)):  # creates the coactor-coactor similarity matrix
    for j in range(0,len(actor_movies)):
        actor1 = set(actor_movies[actor_list[i]])
        actor2 = set(actor_movies[actor_list[j]])
        count_set = actor1.intersection(actor2)
        co_co[i][j] = len(count_set)
        co_co[j][i] = len(count_set)
        

#print(co_co)

U, s, V = linalg.svd(co_co)  # SVD implementation

latent_semantics = pd.DataFrame({'actorid':actor_list})
latent_semantics['latent_semantic_1'] = U[0]
latent_semantics['latent_semantic_2'] = U[1]
latent_semantics['latent_semantic_3'] = U[2]

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
    print("Group " + str(i+1))
    print(grouping_df.iloc[i]['actorid'])
