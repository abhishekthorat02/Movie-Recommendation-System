
import pandas as pd
import numpy as np
from sktensor import dtensor, cp , ktensor, sptensor
from sklearn.cluster import KMeans
#import tensorly as td


def load():
    ###### LOAD DATA FOR TENSOR #########
    actor_df = pd.read_csv("task1/database/phase1_dataset/imdb-actor-info.csv")
    actor_list = list(set(actor_df['id']))

    movies_df = pd.read_csv("task1/database/phase1_dataset/movie-actor.csv")

    movie_list = list(set(movies_df['movieid']))
    year_df = pd.read_csv("task1/database/phase1_dataset/mlmovies.csv")
    year_list = list(set(year_df['year']))
    
    movie_year = pd.merge(movies_df, year_df, on = 'movieid')
    movie_year = movie_year.drop('genres', axis = 1)
    movie_year = movie_year.drop('actor_movie_rank', axis = 1)
    
    actor_movie_year = pd.merge(movie_year, actor_df, left_on = 'actorid', right_on = 'id')
    actor_movie_year = actor_movie_year.drop(['id','moviename','name','gender'], axis = 1)

    amy_triple = actor_movie_year
    amy_triple['value'] = 1 # valid triples set to 1
     
    movies = np.array(movie_list)
    actors = np.array(actor_list)
    years = np.array(year_list)
    
    mat = np.zeros((len(actors), len(movies), len(years))) # 3D Array
    
    tensor_df = pd.Panel(mat, items = actor_list, major_axis = movie_list, minor_axis = year_list) # Creates a Tensor
    tensor_df = tensor_df.to_frame(filter_observations = False)
    
    for row in range(0,len(amy_triple)): #Initialize the tensor
        tensor_df.loc[(amy_triple.iloc[row]['movieid'], amy_triple.iloc[row]['year']),amy_triple.iloc[row]['actorid']] = 1
   

    m,n = len(tensor_df.index.levels[0]), len(tensor_df.index.levels[1])
    arr = tensor_df.values.reshape(m,n,-1).swapaxes(1,2)
    arr = arr.swapaxes(0,1)
    
    return (arr,movies,actors,years)

def display(actor, movies, year):
    ### Function to display the latent semantics ###
    print("Latent semantics for actors")
    for i in range(0,5):
        print("Latent Semantic" + str(i+1))
        print(actors[['actorid','latent_semantic_'+str(i+1)]].sort_values(['latent_semantic_'+str(i+1)], ascending = False))
        
    print("Latent semantics for movies")
    for i in range(0,5):
        print("Latent Semantic" + str(i+1))
        print(movies[['movieid','latent_semantic_'+str(i+1)]].sort_values(['latent_semantic_'+str(i+1)], ascending = False))

    print("Latent semantics for years")
    for i in range(0,5):
        print("Latent Semantic" + str(i+1))
        print(years[['year','latent_semantic_'+str(i+1)]].sort_values(['latent_semantic_'+str(i+1)], ascending = False))

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



tensor,m,a,y = load() 

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
actors = pd.DataFrame(a,columns = ['actorid'])
movies = pd.DataFrame(m,columns = ['movieid'])
years = pd.DataFrame(y,columns = ['year'])

### COMBINE CORRESPONING MODES WITH FACTOR MATRICES ###
for i in range(0,5): 
    actors['latent_semantic_'+ str(i+1)] = Factor1[:,i]
    movies['latent_semantic_'+ str(i+1)] = Factor2[:,i]
    years['latent_semantic_'+ str(i+1)] = Factor3[:,i]

display(actors, movies, years) # CALL TO DISPLAY

clustering(actors,a) # CALL TO KMEANS FUNCTION
clustering(movies,m)
clustering(years,y)

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

#plt.plot(Factor3[:,0])
#plt.show()

#plt.plot(Factor3[:,1])
#plt.show()
#plt.plot(Factor3[:,2])
#plt.show()
plt.plot(Factor3[:,3])
plt.show()
plt.plot(Factor3[:,4])
plt.show()


'''

