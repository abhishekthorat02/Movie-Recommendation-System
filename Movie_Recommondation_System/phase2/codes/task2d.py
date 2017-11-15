import pandas as pd
import numpy as np
#import tensorly as tl
from sktensor import dtensor, cp , ktensor, sptensor
from sklearn.cluster import KMeans

def load():
    ### LOAD DATA FOR TENSOR ###
    mlratings = pd.read_csv("task1/database/phase1_dataset/mlratings.csv")
    genome_tags = pd.read_csv("task1/database/phase1_dataset/genome-tags.csv")

    rating_list = list(set(mlratings['rating']))
    movie_list = list(set(mlratings['movieid']))
    tag_list = list(set(genome_tags['tagId']))

    mlratings = mlratings.drop(['imdbid','timestamp'], axis = 1)
    movie_ratings = mlratings.groupby('movieid',as_index = False)['rating'].mean()
    movie_ratings = movie_ratings.rename(columns = {"rating" : "Avg_rating"})

    rel = pd.merge(movie_ratings, mlratings, on = 'movieid')
    rel = rel[rel.rating >= rel.Avg_rating] #Only users who gave higher rating than the average are kept

    mltags = pd.read_csv("task1/database/phase1_dataset/mltags.csv")
    mltags = mltags.drop(['timestamp'], axis = 1)

    triple = pd.merge(rel, mltags, on = ['userid','movieid'], how = 'inner') # VALID TRIPLES
    triple['value'] = 1

    movies = np.array(movie_list)
    tags = np.array(tag_list)
    ratings = np.array(rating_list)
    m = len(movies)
    t = len(tags)
    r = len(ratings)
    mat = np.zeros((len(tags), len(movies), len(ratings))) # 3D ARRAY

    tensor_df = pd.Panel(mat, items = tag_list, major_axis = movie_list, minor_axis = rating_list)
    tensor_df = tensor_df.to_frame(filter_observations = False)

    for row in range(0,len(triple)): # TENSOR INITIALIZED
        tensor_df.loc[(triple.iloc[row]['movieid'], triple.iloc[row]['rating']), triple.iloc[row]['tagid']] = 1

    m,n = len(tensor_df.index.levels[0]), len(tensor_df.index.levels[1])
    arr = tensor_df.values.reshape(m,n,-1).swapaxes(1,2)
    arr = arr.swapaxes(0,1)

    return(arr,movies,tags,ratings)

def display(tags, movies, ratings):
    ### FUNCTION TO DISPLAY LATENT SEMANTICS ###
    print("Latent semantics for tags")
    for i in range(0,5):
        print("Latent Semantic" + str(i+1))
        print(tags[['tagid','latent_semantic_'+str(i+1)]].sort_values(['latent_semantic_'+str(i+1)], ascending = False))
        
    print("Latent semantics for movies")
    for i in range(0,5):
        print("Latent Semantic" + str(i+1))
        print(movies[['movieid','latent_semantic_'+str(i+1)]].sort_values(['latent_semantic_'+str(i+1)], ascending = False))

    print("Latent semantics for ratings")
    for i in range(0,5):
        print("Latent Semantic" + str(i+1))
        print(ratings[['rating','latent_semantic_'+str(i+1)]].sort_values(['latent_semantic_'+str(i+1)], ascending = False))

def clustering(latent_semantics,ids):
    ### Function to cluster the objects into different groupings based using KMEANS according to the latent semantics ###
    latent_semantics = latent_semantics.drop(latent_semantics.columns[0], axis = 1)
    samples = latent_semantics.as_matrix()
    #print(samples)
    kmeans = KMeans(n_clusters = 5)
    cluster = kmeans.fit_predict(samples)
    #print(cluster)

    grouping_df = pd.DataFrame({'id':ids})
    grouping_df['cluster_id'] = cluster

    grouping_df = grouping_df.groupby('cluster_id',as_index = False).agg(lambda x: list(x))
    print(grouping_df)

    for i in range(0,len(grouping_df)):
        print("Group "+str(i+1))
        print(grouping_df.iloc[i]['id'])


tensor,m,t,r = load()

T = dtensor(tensor) # SCIKIT TENSOR REPRESENTATION

### CP-DECOMPOSITION ###
P, fit, itr, exectimes = cp.als(T, 5, init = 'random')

### FACTOR MATRICES GENERATED ###
Factor1 = P.U[0]
Factor2 = P.U[1]
Factor3 = P.U[2]

### LAMBDA VALUES ###
'''
lamda1 = P.lmbda[0]
lamda2 = P.lmbda[1]
lamda3 = P.lmbda[2]
'''

tags = pd.DataFrame(t,columns = ['tagid'])
movies = pd.DataFrame(m,columns = ['movieid'])
ratings = pd.DataFrame(r,columns = ['rating'])

### COMBINE CORRESPONDING FACTOR MATRICES ###
for i in range(0,5): 
    tags['latent_semantic_'+ str(i+1)] = Factor1[:,i]
    movies['latent_semantic_'+ str(i+1)] = Factor2[:,i]
    ratings['latent_semantic_'+ str(i+1)] = Factor3[:,i]

display(tags, movies, ratings)  # CALL TO DISPLAY

clustering(tags,t) # CALL TO KMEANS FUNCTION
clustering(movies,m)
clustering(ratings,r)

'''
print "Factor1"
print Factor1

print "lamda1"
print lamda1


print "Factor2"
print Factor2

print "lamda2"
print lamda2


print "Factor3"
print Factor3

print "lamda3"
print lamda3

print fit
print itr
print exectimes

print P.lmbda
'''
