[staticmethod]

def combine_map_value_list(input_map):
    combined_list = []
    for value in input_map.values():
        combined_list += value
    return combined_list

[staticmethod]
def fill_matrix(matrix, rows, columns, tfidf_map):
    for pair in tfidf_map:
        for key, value in pair[1].iteritems():
            matrix[rows.index(pair[0])][columns.index(key)] = value

    return matrix

[staticmethod]
def fill_matrix_count(matrix, map, rows, columns):
    for key in map:
        for value in map[key]:
            matrix[rows.index(key)][columns.index(value)] = map[key].count(value)
    return matrix
