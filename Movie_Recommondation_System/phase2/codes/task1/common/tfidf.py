#!/usr/bin/env python

import sys
import os
import math
import copy
from common import basics
from common import createDataSet
import numpy as np
import itertools

class TfIdf:
    def __init__(self):
        self.weighted = False
        self.tf = []
        self.tfidf = []
        self.corpus = set()
        self.idf_map = {}
        self.data_set = createDataSet.CreateDataSet()

    def calculate_tf(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus.add(w)

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.tf.append([doc_name, doc_dict])

    def calculate_idf(self):
        for elem in self.corpus:
            count = 0
            for document in self.tf:
                if elem in document[1].keys():
                    count = count + 1.0

            # calculate IDF
            self.idf_map[elem] = math.log(len(self.tf)/count, 10)


    def calculate_tfidf(self):
        self.calculate_idf()
        self.tfidf = copy.deepcopy(self.tf)
        for index in range(len(self.tf)):
            for key in self.tf[index][1]:
                self.tfidf[index][1][key] = self.tf[index][1][key] * self.idf_map[key]

    def cosine_similarity(self, vector1, vector2):
        dot_product = 0
        keys = set(vector1.keys()+vector2.keys())

        for key in keys:
            dot_product = dot_product + vector1.get(key, 0) * vector2.get(key, 0)

        magnitude = math.sqrt(sum([val ** 2 for val in vector1.values()])) * math.sqrt(sum([val ** 2 for val in vector2.values()]))
        if not magnitude:
            return 0
        return dot_product / magnitude

    def manhattan_distance(self, vector1, vector2):
        keys = set(vector1.keys()+vector2.keys())
        distance = 0

        for key in keys:
            distance += abs(vector1.get(key, 0) - vector2.get(key, 0))
        return distance

    def calculate_timeweighted_tf(self, input_map, normalized_timestamp):
        tag_counts = []
        row = 0
        count = 0
        for movie in input_map:
            tag_counts.append(len(input_map[movie]))

        for tag, weight in zip(basics.combine_map_value_list(input_map), normalized_timestamp):
            self.tf[row][1][tag] += weight
            count = count + 1
            if tag_counts[row] == count:
                row = row + 1
                count = 0

    def calculate_pdiff1(self,genre1, genre2):
        movie_tag_map1, tagid_tag_map1, movies_genre1 = self.data_set.get_movie_tag_by_genre(genre1)
        movie_tag_map2, tagid_tag_map2, movies_genre2 = self.data_set.get_movie_tag_by_genre(genre2)
        tags = tagid_tag_map1.values()
        total_movies_genre1 = float(len(movies_genre1))
        total_movies = float(len(set(movies_genre1 + movies_genre2)))
        tag_weight_genre1 = {}
        # Total Set
        movie_tag_both = movie_tag_map1
        movie_tag_both.update(movie_tag_map2)

        # calculate number of movies in genre, x, containing the tag t
        for tag in tags:
            numbers_tags_genre1 = 0.0
            numbers_tags_genre2 = 0.0
            for movie in movie_tag_map1:
                numbers_tags_genre1 += tag in movie_tag_map1[movie]
            for movie in movie_tag_both:
                numbers_tags_genre2 += tag in movie_tag_both[movie]

            first_entity = math.log((((numbers_tags_genre1+0.5) / (total_movies_genre1 - numbers_tags_genre1 + 0.5)) / (
            abs((numbers_tags_genre2 - numbers_tags_genre1+0.5)) / (
            total_movies - numbers_tags_genre2 - total_movies_genre1 + numbers_tags_genre1 + 0.5))), 10)


            second_entity = abs((numbers_tags_genre1 / total_movies_genre1) - (
            (numbers_tags_genre2 - numbers_tags_genre1) / (total_movies - total_movies_genre1)))

            tag_weight_genre1[tag] = abs(first_entity) * second_entity

        return tag_weight_genre1

    def calculate_pdiff2(self,genre1, genre2):
        movie_tag_map1, tagid_tag_map1, movies_genre1 = self.data_set.get_movie_tag_by_genre(genre1)
        movie_tag_map2, tagid_tag_map2, movies_genre2 = self.data_set.get_movie_tag_by_genre(genre2)
        tags = tagid_tag_map1.values()
        total_movies_genre2 = float(len(movies_genre2))
        total_movies = float(len(set(movies_genre1 + movies_genre2)))
        tag_weight_genre1 = {}
        # Total Set
        movie_tag_both = movie_tag_map1
        movie_tag_both.update(movie_tag_map2)
        # calculate number of movies in genre, x, containing the tag t
        for tag in tags:
            numbers_tags_not_genre2 = 0.0
            numbers_tags_not_genre12 = 0.0
            for movie in movie_tag_map2:
                numbers_tags_not_genre2 += tag not in movie_tag_map2[movie]
            for movie in movie_tag_both:
                numbers_tags_not_genre12 += tag not in movie_tag_both[movie]


            first_entity = math.log((((numbers_tags_not_genre2+0.5) / (total_movies_genre2 - numbers_tags_not_genre2 + 0.5)) / (
            abs((numbers_tags_not_genre12 - numbers_tags_not_genre2+0.5)) / (
            total_movies - numbers_tags_not_genre12 - total_movies_genre2 + numbers_tags_not_genre2+0.5))), 10)

            second_entity = abs((numbers_tags_not_genre2 / total_movies_genre2) - (
            (numbers_tags_not_genre12 - numbers_tags_not_genre2) / (total_movies - total_movies_genre2)))

            tag_weight_genre1[tag] = abs(first_entity) * second_entity

        return tag_weight_genre1

    def get_actors_tags_space(self):
        # variable init
        actor_tag_map = {}
        actor_rank_map = {}
        actor_timestamp_map = {}
        data_set = self.data_set
        tfidf = TfIdf()

        # PART 1 - get complete data set

        # Get all the tags for movies for all actors


        actor_tag_query = "select actorid, tagid, timestamp,actor_movie_rank from (0!.moviedata.movie_actor) ij `movieid xgroup .moviedata.mltags"

        # Execute Query

        movie_tag_data = data_set.get_data_from_kdb(actor_tag_query)

        # Add  tag-movie and tag-timestamp relationship to map

        for elem in movie_tag_data:
            if(elem[0] not in actor_tag_map.keys()):
                actor_tag_map[elem[0]] = np.array(elem[1]).tolist()
                actor_timestamp_map[elem[0]] = np.array(elem[2]).tolist()
                actor_rank_map[elem[0]] = len(np.array(elem[2]).tolist()) * np.array(elem[3]).tolist()
            else:
                actor_tag_map[elem[0]] += np.array(elem[1]).tolist()
                actor_timestamp_map[elem[0]] += np.array(elem[2]).tolist()
                actor_rank_map[elem[0]] += len(np.array(elem[2]).tolist()) * np.array(elem[3]).tolist()

        # PART 3 - TF IDF

        # Calculate TF per document
        for key in actor_tag_map:
            tfidf.calculate_tf(key, actor_tag_map[key])

        # Normalize data and find time weighted TF under timestamp tag

        rank_data = {'rank': basics.combine_map_value_list(actor_timestamp_map)}
        df = data_set.normalize_data(rank_data)
        normalized_rank = df['rank'].tolist()

        # Calculate time weighted TF
        tfidf.calculate_timeweighted_tf(actor_tag_map, normalized_rank)

        # Calculate TF-IDF for all documents
        tfidf.calculate_tfidf()
        # calculate all tags and all movies
        alltags = basics.combine_map_value_list(actor_tag_map)
        allactors = actor_tag_map.keys()

        # create object feature matrix with zero values
        object_feature_matrix = np.zeros(shape=(len(allactors), len(alltags)))
        # fill up the matrix
        object_feature_matrix = basics.fill_matrix(object_feature_matrix, allactors, alltags, tfidf.tfidf)
        return(object_feature_matrix, allactors, alltags, actor_tag_map)
