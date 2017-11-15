from databaseConnectionKDB import connect_to_kdbdb
import logging
import pandas as pd
import numpy as np
import basics

class CreateDataSet:

    def __init__(self):
        # connect To DB
        self.kdb_conn = connect_to_kdbdb()
        # create logger
        self.log = logging.getLogger("MWDB")

    def get_data_from_kdb(self, query):

        # Get data from KDB database
        try:
            data = self.kdb_conn(query)
            return [elem for elem in data]
        except:
            self.log.error("ERROR-FETCH-KDB: Unable To fetch data for query - " + query)

    def get_symbol_list(self,input_list):
        try:
            return ";".join(input_list)
        except:
            self.log.error("Can not convert to Symbol")

    def nparray_to_list(self, nparray):
        try:
            return np.array(nparray).tolist()
        except :
            self.log.error("ERROR: can not convert to list")

    def get_all_indicies(self, big_list, small_list):
        indicies = []
        for elem in small_list:
            try:
                indicies.append(big_list.index(elem))
            except:
                continue
        return indicies

    def normalize_data(self,input_map):
        df = pd.DataFrame(input_map)
        df_norm = (df - df.min()) / (df.max() - df.min())
        return df_norm

    def get_actors_by_movie(self,input_movie):
        movie_actor_query = 'exec actorid from .moviedata.movie_actor where movieid=`$"' + input_movie + '"'
        # Execute Query
        genre_movie_list = self.get_data_from_kdb(movie_actor_query)
        movie_actors = np.array(genre_movie_list).tolist()
        return movie_actors

    def get_movies_by_genre(self,input_genre):
        genre_movies = []
        for genre in input_genre:
            genre_movie_query = 'exec movieid from .moviedata.mlmovies where string[genres] like "' + genre + '"'
            # Execute Query
            genre_movie_list = self.get_data_from_kdb(genre_movie_query)
            genre_movies += np.array(genre_movie_list).tolist()
        return genre_movies

    def get_movie_tag_by_genre(self,input_genre,timestamp_map=False):
        movie_tag_map = {}
        tagid_tag_map = {}
        movie_timestamp_map = {}
        movie_genre_query = 'exec movieid from .moviedata.mlmovies where genres=`$"' + input_genre + '"'

        # Execute Query

        movies = np.array(self.get_data_from_kdb(movie_genre_query)).tolist()

        # PART 2 - get complete data set

        sym = self.get_symbol_list(movies)

        # Get all the tags for movies for give genre

        movie_tag_query = "select movieid,tagid,tag,timestamp from (`movieid xgroup select  movieid,tagid,tag:tagid.tag,timestamp from .moviedata.mltags) where movieid in `$string(" + sym + ")"

        # Execute Query

        movie_tag_data = self.get_data_from_kdb(movie_tag_query)

        # Add  tag-movie and tag-timestamp relationship to map

        for elem in movie_tag_data:
            movie_tag_map[elem[0]] = np.array(elem[2]).tolist()
            movie_timestamp_map[elem[0]] = np.array(elem[3]).tolist()
            tagid_tag_map.update(dict(zip(np.array(elem[1]).tolist(), np.array(elem[2]).tolist())))

        if timestamp_map:
            return [movie_tag_map, tagid_tag_map, movie_timestamp_map, movies]
        else:
            return [movie_tag_map, tagid_tag_map, movies]



