import numpy as np
import xxhash

class LH():
    def __init__(self, domain, eps):
        self.g = int(np.exp(eps)) + 1
        self.p = np.exp(eps) / (np.exp(eps) + self.g - 1)
        self.q = 1 / self.g
        self.domain = domain

    def lh(self, real_dist):
        domain = len(real_dist)    
        noisy_samples = self.lh_perturb(real_dist)
        est_dist = self.lh_aggregate(noisy_samples)
        return est_dist

    def lh_maxgain_perturb(self, real_dist, targets):
        g, p, q = self.g, self.p, self.q
        print("lh max-gain perturbing...")
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        counter = 0
        for k, v in enumerate(real_dist):
            if v == 0 : continue
            seed = 0
            gain = 0
            # randomly sample 100 seeds to attain approximately max gain
            for _ in range(100):
                rseed = np.random.randint(0, n)
                cur_gain = 0
                x = xxhash.xxh32(str(int(k)), seed=rseed).intdigest() % g
                # gain = num of targets supported by tuple (k,rseed)
                for t in targets:
                    temp = xxhash.xxh32(str(int(t)), seed=rseed).intdigest() % g
                    if temp == x:
                        cur_gain += 1
                if cur_gain > gain:
                    gain = cur_gain
                    seed = rseed
            y = xxhash.xxh32(str(int(k)), seed=seed).intdigest() % g
            for _ in range(v):
                noisy_samples[counter] = tuple([y, seed])
                counter += 1
        return noisy_samples

    def lh_perturb(self, real_dist, rand_on = 1):
        g, p, q = self.g, self.p, self.q
        print("lh perturbing...")
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        samples_one = np.random.random_sample(n)
        seeds = np.random.randint(0, n, n)

        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                y = x = xxhash.xxh32(str(int(k)), seed=seeds[counter]).intdigest() % g
                # randomize the output at probability '1-p'
                if rand_on and samples_one[counter] > p:
                    y = np.random.randint(0, g - 1)
                    if y >= x:
                        y += 1
                noisy_samples[counter] = tuple([y, seeds[counter]])
                counter += 1
        return noisy_samples


    def lh_aggregate(self, noisy_samples):
        g, p, q, domain = self.g, self.p, self.q, self.domain
        n = len(noisy_samples)
        print("lh aggregating...")
        print("len(noisy_samples) = ", n, "domain = ", domain)
        est = np.zeros(domain, dtype=np.int32)
        for i in range(n):
            if(i%(int(n/20))==0):
                print("%.2f" %(i*1.0/n))
            for v in range(domain):
                x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
                if noisy_samples[i][0] == x:
                    est[v] += 1

        a = 1.0 / (p - q)
        b = n * q / (p - q)
        est = a * est - b

        return est

class RR():
    def __init__(self, domain, eps):
        self.eps = eps
        self.domain = domain
        self.ee = np.exp(eps)
        self.p = self.ee / (self.ee + domain - 1)
        self.q = 1 / (self.ee + domain - 1)

    def rr(self, real_dist):
        noisy_samples = self.rr_perturb(real_dist)
        est_dist = self.rr_aggregate(noisy_samples)
        return est_dist


    def rr_perturb(self, real_dist):
        print("rr perturbing...")
        p, domain = self.p, self.domain
        n = sum(real_dist)
        perturbed_datas = np.zeros(n, dtype=np.int)
        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                y = x = k
                p_sample = np.random.random_sample()

                if p_sample > p:
                    y = np.random.randint(0, domain - 1)
                    if y >= x:
                        y += 1
                perturbed_datas[counter] = y
                counter += 1
        return perturbed_datas


    def rr_aggregate(self,noisy_samples):
        n = len(noisy_samples)
        p, q, domain = self.p, self.q, self.domain
        print("rr aggregating...")
        est = np.zeros(domain)
        unique, counts = np.unique(noisy_samples, return_counts=True)
        for i in range(len(unique)):
            est[unique[i]] = counts[i]

        a = 1.0 / (p - q)
        b = n * q / (p - q)
        est = a * est - b

        return est
