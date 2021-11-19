import pickle
from os import path
from atk import Adversary
import heapq

import numpy as np


class Data(object):
    def __init__(self, dataname, limit=-1):
        self.data = None
        self.name = dataname
        if dataname == 'kosarak':
            self.dict_size = 42178
            user_total = 990002
        elif dataname == 'IBM':
            self.dict_size = 1000
            user_total = 1800000
        self.user_total = user_total
        random_map = np.arange(user_total)
        np.random.shuffle(random_map)
        overall_count = 0
        user_file_name = '%s-data/%s' % (dataname, dataname)
        print("Loading dataset %s-data..." %(dataname))
        if not path.exists(user_file_name + '.pkl'):
            print("No existence of %s.pkl, loading data manually..." %(user_file_name))
            data = [0] * user_total
            f = open(user_file_name + '.dat', 'r')
            print(f)
            for line in f:
                if len(line) == 0:
                    break
                if line[0] == '#':
                    continue
                queries = line.split(' ')

                data[random_map[overall_count]] = [0] * len(queries)
                for i in range(len(queries)):
                    if dataname == 'IBM' and i < 3:
                        continue
                    query = int(queries[i])
                    data[random_map[overall_count]][i] = query
                data[random_map[overall_count]].sort()
                overall_count += 1
                if overall_count >= user_total:
                    break
            pickle.dump(data, open(user_file_name + '.pkl', 'wb'))
        self.data = pickle.load(open(user_file_name + '.pkl', 'rb'))
        print("type(data) = ", type(self.data))
        print("len(data) = ", len(self.data))
        # only use part of the data to get results quickly
        if limit>0:
            self.data = self.data[:limit]
    
    def true_freq(self, top_k, data):
        """ get true freq of top-k items """
        results = np.zeros(self.dict_size,dtype=np.int)
        for i in range(len(data)):
            for j in range(len(data[i])):
                value = self.data[i][j]
                results[value] += 1
        result = heapq.nlargest(top_k, range(len(results)), results.take)
        result.reverse()
        print("top-k items = ", result)
        return result
    
