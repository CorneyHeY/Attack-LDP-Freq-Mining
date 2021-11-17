import heapq
import time
import math
import numpy as np
import fo
from atk import Adversary

class SVSM():
    use_atk = True
    phase1_percent = 0.4
    phase2_percent = 0.1
    phase3_percent = 0.5
    def __init__(self, data, attacker, top_k, epsilon):
        self.data = data
        self.atk = attacker
        self.top_k = top_k
        self.epsilon = epsilon

    def find(self,mode):
        n = len(self.data.data)
        single_test_user = int(1 * n)

        key_list, est_freq = self.find_singleton(single_test_user)
        if mode == 0:
            cand_items = dict(zip(key_list, est_freq))
            true_results = self.data.true_freq(top_k=self.top_k,data=self.data.data[0:single_test_user])
            sum_utilities = 0
            est_num = 0
            for i in range(self.top_k):
                if key_list[i] in true_results:
                    est_num +=1
                    sum_utilities += true_results.index(key_list[i]) + 1
            accuracy = est_num / self.top_k
            ncr = sum_utilities / ((self.top_k + 1) * self.top_k / 2.0)
            print("accuracy = ",accuracy)
            print("ncr = ",ncr)
            # write into log file
            log_time = time.strftime("%Y%m%d_%H%M", time.localtime()) 
            result_file_name = '%s_%s-result_%s.txt' % (log_time, self.data.name, ("atk" if self.use_atk else "no_atk"))
            fp = open(result_file_name, "w")
            fp.write("#%s dataset#\n" %(self.data.name))
            fp.write("num_of_transactions = %d, num_of_categories = %d\n" %(self.data.user_total,self.data.dict_size))
            fp.write("FreqItemMining involved data - %d\n" % (single_test_user))
            fp.write("Attacker involved size - %d\n" % (int)(single_test_user*self.atk.atk_percent))
            #fp.write("Phase user allocation - %.2f:%.2f:%.2f\n" % (self.phase1_percent, self.phase2_percent,, self.phase3_percent,))
            if self.use_atk:
                fp.write("Attacker/Users percent  - %.2f\n" % (self.atk.atk_percent))
                fp.write("Attacking set = %s\n" % (self.atk.P1_atk_list))
                fp.write("Phase1 Attack success = %s\n" % (self.atk.success_cand_list))
                fp.write("Phase3 Attack success = %s\n" % (self.atk.success_item_list))
                fp.write("Attack increment = %s\n" % (self.atk.atk_items_increment))
                fp.write("Final attacking result = %s\n" % (self.atk.atk_items))
                fp.write("Final result = %s\n" % (cand_items))
            fp.write("accuracy = %f\n" % (accuracy))
            fp.write("ncr = %f\n" % (ncr))
            return cand_items
        else:
            cand_itemsets = self.find_itemset(key_list, est_freq, single_test_user, n)
            return cand_itemsets

    def find_singleton(self, single_test_user):
        p1_user = int(single_test_user * self.phase1_percent)
        p2_user = p1_user + int(single_test_user * self.phase2_percent)
        p3_user = p2_user + int(single_test_user * self.phase3_percent)
        phase1_user = self.data.data[0:p1_user]
        phase2_user = self.data.data[p1_user+1:p2_user]
        phase3_user = self.data.data[p2_user+1:p3_user]

        print("precentage of phase1_user = %.2f" %(len(phase1_user)*1.0/single_test_user))
        print("precentage of phase2_user = %.2f" %(len(phase2_user)*1.0/single_test_user))
        print("precentage of phase3_user = %.2f" %(len(phase3_user)*1.0/single_test_user))
        # step 1: find singleton candidate set
        print("Phase1: find singleton candidate set")
        phase1_atk = self.atk.P1_atk(phase1_user)
        #phase1_data = phase1_user+phase1_atk if self.use_atk else phase1_user
        true_user_dist = self.test_single(phase1_user)
        true_atk_dist = self.test_single(phase1_atk)
        est_user_dist = fo.lh(true_user_dist, self.epsilon)
        est_atk_dist = fo.lh(true_atk_dist, self.epsilon)
        est_singleton_dist = []
        for i in range(len(est_user_dist)):
            est_singleton_dist.append(est_user_dist[i] + est_atk_dist[i])
        top_singleton = 2 * self.top_k
        singleton_list, value_result = self.build_result(est_singleton_dist, range(len(est_singleton_dist)),
                                                         top_singleton)
        self.atk.P1_assess(singleton_list,est_user_dist,est_atk_dist)

        # step 2: find an appropriate length
        print("Phase2: find an appropriate length")
        phase2_atk = self.atk.P2_atk(phase2_user)
        phase2_data = phase2_user+phase2_atk if self.use_atk else phase2_user
        key_result = {}
        for i in range(len(singleton_list)):
            key_result[(singleton_list[i],)] = i

        length_percentile = 0.9
        length_distribution = self.test_length_singleton(phase2_data, len(singleton_list),
                                                         singleton_list)
        length_limit = self.find_percentile_set(length_distribution, length_percentile)

        # step 3: test with the confined set
        print("Phase3: test with the confined set")
        phase3_atk = self.atk.P3_atk(phase3_user)
        #phase3_data = phase3_user+phase3_atk if self.use_atk else phase3_user
        use_grr, eps = self.set_grr(key_result, length_limit)
        true_user_dist = self.test_singleton_cand_limit(phase3_user, key_result, set(singleton_list), length_limit)
        true_atk_dist = self.test_singleton_cand_limit(phase3_atk, key_result, set(singleton_list), length_limit)
        
        if use_grr:
            user_value_estimates = fo.rr(true_user_dist, eps)[:-1]
            atk_value_estimates = fo.rr(true_atk_dist, eps)[:-1]
        else:
            user_value_estimates = fo.lh(true_user_dist, eps)[:-1]
            atk_value_estimates = fo.lh(true_atk_dist, eps)[:-1]
        value_estimates = np.array(user_value_estimates)
        for i in range(len(user_value_estimates)):
            value_estimates[i] += atk_value_estimates[i]
        total_user = single_test_user+len(phase1_atk)+len(phase2_atk)+len(phase3_atk)
        factor = total_user / length_percentile / (len(phase3_user)+len(phase3_atk))
        value_estimates *= total_user / length_percentile / (len(phase3_user)+len(phase3_atk))

        top_singleton = self.top_k
        key_list, est_freq = self.build_result(value_estimates, singleton_list, top_singleton)

        self.atk.P3_assess(key_list, user_value_estimates, value_estimates, singleton_list)

        return key_list, est_freq

    def find_itemset(self, singletons, singleton_freq, single_test_user, multi_test_user):

        # step 1: build itemset candidate
        set_cand_dict, set_cand_dict_inv = self.get_set_cand_thres_prod(singletons, singleton_freq)

        # step 2: itemset size distribution
        length_percentile = 0.9
        percentile_test_user = single_test_user + int(0.2 * (multi_test_user - single_test_user))
        length_distribution_set = self.test_length_itemset(single_test_user + 1, percentile_test_user,
                                                           len(set_cand_dict), set_cand_dict)
        length_limit = self.find_percentile_set(length_distribution_set, length_percentile)

        # step 3: itemset est
        true_itemset_dist = self.data.test_cand_limit(percentile_test_user + 1, multi_test_user, set_cand_dict,
                                                      length_limit)
        use_grr, eps = self.set_grr(true_itemset_dist, length_limit)

        if use_grr:
            set_freq = fo.rr(true_itemset_dist, eps)[:-1]
        else:
            set_freq = fo.lh(true_itemset_dist, eps)[:-1]
        set_freq *= single_test_user / length_percentile / (multi_test_user - percentile_test_user)

        self.update_tail_with_reporting_set(length_limit, length_distribution_set, set_cand_dict, set_freq)

        return self.build_set_result(singletons, singleton_freq, set_freq, set_cand_dict_inv)

    # ===== singleton testing methods
    def test_single(self, data):
        """ every user report a random item from all its items with no padding """
        results = np.zeros(self.data.dict_size, dtype=np.int)
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
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = key_dict[(values[rand_index],)]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(singleton_set)
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

    # ===== auxiliary functions for singletons
    def build_result(self, value_estimates, key_list, top_singleton):
        sorted_indices = np.argsort(value_estimates)
        key_result = []
        value_result = []
        for j in sorted_indices[-top_singleton:]:
            key_result.append(key_list[j])
            value_result.append(value_estimates[j])

        return key_result, value_result

    def find_percentile_set(self, length_distribution, length_percentile):
        total = sum(length_distribution)
        current_total = 0
        for i in range(1, len(length_distribution)):
            if i > 30:
                break
            current_total += length_distribution[i]
            if current_total / total > length_percentile:
                break
        return i

    def set_grr(self, new_cand_dict, length_limit):
        eps = self.epsilon
        use_grr = False
        if len(new_cand_dict) < length_limit * math.exp(self.epsilon) * (4 * length_limit - 1) + 1:
            eps = math.log(length_limit * (math.exp(self.epsilon) - 1) + 1)
            use_grr = True
        return use_grr, eps

    def test_length_singleton(self, user_data, length_limit, cand_dict):
        true_length_dist = self.test_length_cand(user_data, cand_dict, length_limit)
        est_length_dist = fo.lh(true_length_dist, self.epsilon)
        return est_length_dist

    # ===== auxiliary functions for itemset: constructing candidate set
    def get_set_cand_thres_prod(self, key_list, est_freq):

        cand_dict = {}
        for i in range(len(key_list)):
            cand_dict[key_list[i]] = est_freq[i]

        normalized_values = np.zeros(len(est_freq))
        for i in range(len(est_freq)):
            normalized_values[i] = (est_freq[i] * 0.9 / est_freq[-1])
        cand_dict = {}
        cand_dict_prob = {}
        new_cand_inv = []
        self.build_tuple_cand_bfs(cand_dict_prob, cand_dict, new_cand_inv, key_list, normalized_values)
        cand_list = list(cand_dict.keys())
        cand_value = list(cand_dict_prob.values())
        sorted_indices = np.argsort(cand_value)
        new_cand_dict = {}
        new_cand_inv = []
        for j in sorted_indices[-self.top_k:]:
            new_cand_dict[cand_list[j]] = len(new_cand_inv)
            new_cand_inv.append(tuple(cand_list[j]))
        return new_cand_dict, new_cand_inv

    def build_tuple_cand_bfs(self, cand_dict_prob, cand_dict, new_cand_inv, keys, values):
        ret = []
        cur = []
        for i in range(len(keys)):
            heapq.heappush(ret, (values[i], tuple([i])))
            heapq.heappush(cur, (-values[i], tuple([i])))
        while len(cur) > 0:
            new_cur = []
            while len(cur) > 0:
                (prob, t) = heapq.heappop(cur)
                for j in range(t[-1] + 1, len(keys)):
                    if -prob * values[j] > ret[0][0]:
                        if len(ret) >= 3 * len(keys):
                            heapq.heappop(ret)
                        l = list(t)
                        l.append(j)
                        heapq.heappush(ret, (-prob * values[j], tuple(l)))
                        heapq.heappush(new_cur, (prob * values[j], tuple(l)))
            cur = new_cur

        while len(ret) > 0:
            (prob, t) = heapq.heappop(ret)
            if len(t) == 1:
                continue
            l = list(t)
            new_l = []
            for i in l:
                new_l.append(keys[i])
            new_t = tuple(new_l)
            cand_dict[new_t] = len(new_cand_inv)
            cand_dict_prob[new_t] = prob
            new_cand_inv.append(new_t)

    def test_length_itemset(self, user_start, user_end, length_limit, cand_dict):
        true_length_dist = self.data.test_length_itemset(user_start, user_end, cand_dict, length_limit)
        est_length_dist = fo.lh(true_length_dist, self.epsilon)
        return est_length_dist

    def update_tail_with_reporting_set(self, length_limit, length_distribution_set, key_result, value_result):
        addi_total_item = 0
        for i in range(length_limit + 1, len(length_distribution_set)):
            addi_total_item += length_distribution_set[i] * (i - length_limit)
            if length_distribution_set[i] == 0:
                break
        total_item = sum(value_result)

        ratio = addi_total_item * 1.0 / total_item
        for i in range(len(value_result)):
            value_result[i] *= (1.0 + ratio)
        return key_result, value_result

    def build_set_result(self, singletons_keys, singleton_freq, set_freq, set_cand_dict_inv):
        current_estimates = np.concatenate((singleton_freq, set_freq), axis=0)
        results = {}
        sorted_indices = np.argsort(current_estimates)
        for j in sorted_indices[-self.top_k:]:
            if j < len(singletons_keys):
                results[tuple([singletons_keys[j]])] = singleton_freq[j]
            else:
                l = list(set_cand_dict_inv[j - len(singletons_keys)])
                l.sort()
                results[tuple(l)] = current_estimates[j]

        return results
