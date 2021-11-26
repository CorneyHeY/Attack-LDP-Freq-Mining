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
    
    # ===== singleton testing methods
    def test_single(self, data):
        """ every user report a random item from all its items with no padding """
        results = np.zeros(self.dict_size, dtype=np.int)
        for i in range(len(data)):
            if len(data[i]) == 0:
                continue
            rand_index = np.random.randint(len(data[i]))
            value = data[i][rand_index]
            results[value] += 1
        return results

    def test_length_cand(self, data, cand_list, limit, start_limit=0):
        """ every user report the length of its intersection with candidate set """
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        cand_set = set(cand_list)
        for i in range(len(data)):
            X = data[i]
            # V = cand_set.intersection(X)
            # value = len(V)
            value = 0
            for item in X:
                if item in cand_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    def test_singleton_cand_limit(self, data, key_dict, singleton_set, length_limit):
        results = np.zeros(len(singleton_set) + 1, dtype=np.int)
        for i in range(len(data)):
            values = [] # set of intersected items 
            x = data[i]
            for item in x:
                if item in singleton_set:
                    values.append(item)
            # pad to L
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = key_dict[(values[rand_index],)]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(singleton_set) # dummy item
                if rand_index < len(values):
                    result = key_dict[(values[rand_index],)]
            results[result] += 1
        return results

    # ===== itemset testing methods
    def test_length_itemset(self, data, cand_dict, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))
        for i in range(len(data)):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            value = 0
            for cand in cand_dict:
                if set(cand) <= current_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    def test_cand_limit(self, data, cand_dict, length_limit):
        buckets = np.zeros(len(cand_dict) + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))

        for i in range(len(data)):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            subset_count = 0
            subset_indices = []
            for cand in cand_dict:
                if set(cand) <= current_set:
                    subset_count += 1
                    subset_indices.append(cand_dict[cand])

            if subset_count > length_limit:
                rand_index = np.random.randint(subset_count)
                result = subset_indices[rand_index]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(cand_dict)
                if rand_index < subset_count:
                    result = subset_indices[rand_index]

            buckets[result] += 1
        return buckets