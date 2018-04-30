import numpy as np
from scipy.stats import skew
import numpy as np
import os
import json
import random

all_functions = [min, max, np.mean, np.std, skew, len]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4*24),
    "first8days": (0, 0, 0, 8*24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
                         (3, 10), (3, 25), (3, 50)]

def get_range(begin, end, period):
    # first p %
    if (period[0] == 2):
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p % 
    if (period[0] == 3):
        return (end - (end - begin) * period[1] / 100.0, end)
    
    if (period[0] == 0):
        L = begin + period[1]
    else:
        L = end + period[1]
    
    if (period[2] == 0):
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if (len(channel_data) == 0):
        return np.full((len(functions,)), np.nan)
    
    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)
            
    data = [x for (t, x) in channel_data 
                if t > L - 1e-6 and t < R + 1e-6]
    
    if (len(data) == 0):
        return np.full((len(list(functions,))), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(list(data_raw[i]), period, sub_period, functions)
                                                for sub_period in sub_periods],
                                         axis=0) 
                            for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)
    

def extract_features(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                        for x in data_raw])

def read_and_extract_features(reader, period, features):
    ret = read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if (len(channel_info[channel]['possible_values']) != 0) and 'values' in channel_info[channel]:
            ret[i-1] = dict_helper(ret[i-1], channel_info, channel)
        
        ret[i-1] = remove_wrong_values(data)
        ret[i-1] = map(lambda x: (float(x[0]), float(x[1])), ret[i-1])
    return ret

def remove_wrong_values(data):
    counter = 0
    for value in data:
        if(type(value[0]) is str):
            value[0] = value[0].replace('>','').replace('/minute','').replace('/min retracts','')
        if(type(value[1]) is str):
            value[1] = value[1].replace('>','').replace('/minute','').replace('/min retracts','')
        if(value[0] == ''):
            value[0] = 'nan'
        if(value[1] == ''):
            value[1] = 'nan'
        data[counter] = value
        counter = counter+1
    return data

def dict_helper(data, channel_info, channel):
    processed_data = []
    for x in data:
        processed_data.append((x[0], channel_info[channel]['values'][x[1]]))
    return processed_data

def extract_features_from_rawdata(chunk, header, period, features):
    with open(os.path.join(os.path.dirname(__file__),"./resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
