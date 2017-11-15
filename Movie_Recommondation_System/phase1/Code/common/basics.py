[staticmethod]

def combine_map_value_list(input_map):
    combined_list = []
    for value in input_map.values():
        combined_list += value
    return combined_list

