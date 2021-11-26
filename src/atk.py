import math
import numpy as np
import xxhash
import fo

class Adversary():
    # 0: no atk 1: chosen_item_atk 2: weak max-gain atk 3: max-gain atk
    atk_mode = 3 
    atk_percent = 0.03
    atk_size = 32
    def __init__(self, data, top_k, epsilon):
        self.data = data
        self.top_k = top_k
        self.epsilon = epsilon

    def P1_atk(self, P1_data):
        result = np.zeros(0, dtype=object)
        if self.atk_mode == 0: return result
        print("attack Phase1 data...")
        size = int(len(P1_data)*self.atk_percent)
        true_top_k_list = self.data.true_freq(self.top_k, P1_data)

        atk_cand_list = [] # randomly chosen items from non-top-k items as attacking items
        while len(atk_cand_list) < self.atk_size:
            rand_num = np.random.randint(self.data.dict_size)
            if true_top_k_list.count(rand_num) == 0:
                atk_cand_list.append(rand_num)
        self.P1_atk_list = atk_cand_list
        print("P1_atk_list = ", atk_cand_list)
        P1_atk_data = []
        for i in range(size):
            P1_atk_data.append(atk_cand_list)

        true_atk_dist = self.data.test_single(P1_atk_data)
        if self.atk_mode == 1:
            LH = fo.LH(len(true_atk_dist), self.epsilon)
            result = LH.lh_perturb(true_atk_dist)
        elif self.atk_mode == 2:
            LH = fo.LH(len(true_atk_dist), self.epsilon)
            result = LH.lh_perturb(true_atk_dist, 0)
        elif self.atk_mode == 3:
            LH = fo.LH(len(true_atk_dist), self.epsilon)
            result = LH.lh_maxgain_perturb(true_atk_dist, atk_cand_list)
        else:
            assert False, "Unknown atk_mode '%d'" % (self.atk_mode)
        return result

    def P1_assess(self, singleton_list, est_user_dist, est_atk_dist):
        if self.atk_mode == 0: return
        success_cand_set = set(singleton_list).intersection(set(self.P1_atk_list))
        self.success_cand_list = list(success_cand_set)
        atk_increment = []
        ori_freq = []
        for i in self.P1_atk_list:
            ori_freq.append(est_user_dist[i])
            atk_increment.append(est_atk_dist[i])
        self.P1_origin_freq = dict(zip(self.P1_atk_list, ori_freq))
        self.P1_atk_increment = dict(zip(self.P1_atk_list, atk_increment))
        print("Phase1 attacked items origin frequency = ", self.P1_origin_freq)
        print("Phase1 attacked items frequency increment = ", self.P1_atk_increment)
        print("Phase1 attack success set = ", success_cand_set)

    def P2_atk(self, P2_data):
        result = []
        return result

    def P3_atk(self, P3_data):
        result = []
        if self.atk_mode == 0: return result
        self.P3_atk_list = []
        for i in range(int(len(self.success_cand_list)/2)):
            self.P3_atk_list.append(self.success_cand_list[i])
        size = int(len(P3_data)*self.atk_percent)
        for i in range(size):
            result.append(self.P3_atk_list)
        return result

    def P3_assess(self, key_list, user_value_estimates, value_estimates, singleton_list):
        if self.atk_mode == 0: return
        success_item_set = set(key_list).intersection(set(self.P1_atk_list))
        self.success_item_list = list(success_item_set)
        args_atk_items = []
        for i in range(len(singleton_list)):
            if self.P1_atk_list.count(singleton_list[i])!=0:
                args_atk_items.append(i)
        key_result = []
        user_value_result = []
        atk_increment = []
        final_value_result = []
        for j in args_atk_items:
            key_result.append(singleton_list[j])
            user_value_result.append(user_value_estimates[j])
            final_value_result.append(value_estimates[j])
            atk_increment.append(value_estimates[j]-user_value_estimates[j])
        self.P3_origin_freq = dict(zip(key_result, user_value_result))
        self.P3_atk_increment = dict(zip(key_result, atk_increment))
        self.P3_final_freq = dict(zip(key_result, final_value_result))
        print("Phase3 attacked items originnal freq = ", self.P3_origin_freq)
        print("Phase3 attacked items freq increment = ", self.P3_atk_increment)
        print("Phase3 attacked items final freq = ", self.P3_final_freq)
        print("Phase3 attack success set = ", success_item_set)