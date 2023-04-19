#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import random
import pandas as pd
import itertools
import math
import time
import concurrent.futures


# In[4]:


class Landscape:
    def __init__(self, smoothness, length=500):

        self.s = smoothness
        self.length = length
        
        total_segments = round(self.length/self.s)
        """Choose set of random points"""
        points = random.choices(range(1, 100), k=total_segments)
        heights = []
        for i in range(total_segments):
            """Smoothen between every two points"""
            a = points[i]
            b = points[(i+1)%total_segments]
            step = np.round((b - a)/self.s, 2)
            for j in range(self.s):
                if a + step*j > 100:
                    heights.append(100)
                else:
                    heights.append(a + step*j)
        self.heights = heights
        """Update in case original length not divisible by smoothness"""
        self.length = len(self.heights)


# In[8]:


def cov_sim(set1, set2):
    """count how many step sizes overlap between agents"""
    count = 0
    for i in set1:
        if i in set2:
            count += 1
    return count


# ### Model

# In[235]:


class Agent():
    def __init__(self, no, h, landscape, sigma=0):
        self.no = no
        self.h = h # Ordered list of step sizes
        self.landscape = landscape
        self.default_start = random.choice(range(landscape.length))
        self.score = 0 # Highest value found
        self.sigma = sigma # Level of noise in signals
        
    def data(self, loc):
        return max(np.random.normal(self.landscape.heights[loc], 
                                    self.sigma), 
                   0.001)
    
    def search(self, start, own_hist, in_hist):
        """own_hist records agent's own search results"""
        """in_hist records known results by the ingroup"""
        start = start # Starting location on landscape
        loc = start # Current location
        findings = own_hist
        if (in_hist[loc] == 0): 
            """Starting loc hasn't been searched by ingroup members"""
            value = self.data(loc)
            findings[loc], maxi = value, value
        else:
            """Starting loc has been searched by ingroup members"""
            maxi = in_hist[loc] # Current max value found
        
        count = 0 # Number of step sizes tried
        n_total = 0
        
        while count < len(self.h):
            nxt = (loc + self.h[n_total%3])%self.landscape.length # Next loc to check
            if in_hist[nxt] == 0:
                """Never checked by ingroup"""
                value = self.data(loc) # Noisy
                findings[nxt], in_hist[nxt] = value, value
                if maxi < value:
                    """Found value higher than current one"""
                    loc, maxi, count = nxt, value, 0
                    n_total += 1
                else:
                    count += 1
                    n_total += 1
            else:
                """Loc already checked by self or ingroup"""
                value = in_hist[nxt]
                if maxi < value:
                    loc, maxi, count = nxt, value, 0
                    n_total += 1
                else:
                    count += 1
                    n_total += 1
        return findings

class Team():
    def __init__(self, members, landscape, trust_level=1):
        self.members = members
        self.landscape = landscape
        self.trust_level = trust_level
        self.trust = dict()
        for a in self.members:
            self.trust[a] = [a]
        if self.trust_level > 0:
            M = self.members
            random.shuffle(M)
            k = math.ceil(len(self.members)*self.trust_level)
            while len(M) > 0:
                subgroup = M[:k]
                M = M[k:]
                for b in subgroup:
                    self.trust[b] = subgroup
    
    def aggregate(self, maps):
        denom = np.sum([np.where(m>0, 1, 0) for m in maps.values()], axis=0)
        num = np.sum([m for m in maps.values()], axis=0)
        return num/(denom + np.where(denom==0, 1,0))
                    
    def tournament(self, start):
        maps = dict() # Cumulative search results by members
        for a in self.members:
            maps[a] = np.array([0]*self.landscape.length)
        on = True
        maxi = 0 # Current max value found
        loc = start # Location where current max value is found

        while on:
            for m in self.members:
                in_hist = np.sum([maps[n] for n in self.trust[m]], axis=0)
                maps[m] = m.search(loc, maps[m], in_hist)
                
            on = False
            new_max, new_loc = np.amax(self.aggregate(maps)), np.argmax(self.aggregate(maps))
            if new_max > maxi:
                on = True # Continue if higher value found in new round
                loc, maxi = new_loc, new_max

        return self.landscape.heights[np.argmax(self.aggregate(maps))]

# In[248]:


class CorrTeam(Team):
    def __init__(self, members, landscape, trust_level=0.5):
        super().__init__(members, landscape, trust_level)
        self.trust = dict()
        for a in self.members:
            self.trust[a] = [a]
        if self.trust_level > 0:
            M = self.members
            k = math.ceil(len(self.members)*self.trust_level)
            while len(M) > 0:
                rmd = random.choice(M)
                M.sort(key=lambda x: cov_sim(rmd.h, x.h), reverse=True)
                subgroup = M[:k]
                for b in subgroup:
                    self.trust[b] = subgroup
                M = M[k:]


# In[307]:

Poolsizes = [12]
Smoothness = range(1, 9)
Sigma = [0]
Trust = [0, 0.33, 0.5, 1]


def run():
    cols = ['smoothness', 'diverse', 'expert', 
            'd_heuristics', 'x_heuristics', 
            'trust', 'sigma', 'poolsize']
    df = pd.DataFrame(columns=cols)
    h_each = 3 # Number of step sizes each agent has
    per_team = 6 # Number of agents per team
    for poolsize in Poolsizes:
        pool = list(range(1, poolsize+1))
        all_perm = list(itertools.permutations(pool, r=h_each)) # All possible step size combo
        for s in Smoothness: # smoothness
            for sigma in Sigma:
                for t in Trust: # trust level
                    np.random.seed()
                    L = Landscape(s)
                    """Select set of random starting points"""
                    starts = random.sample(range(L.length), 100)
                    """Find experts"""
                    agents = [] # all possible heuristic profiles
                    for i in range(len(all_perm)):
                        agents.append(Agent(i, all_perm[i], L, sigma))
                    for a in agents:
                        scores = [] 
                        for i in starts:
                            results = a.search(i, np.array([0]*a.landscape.length), np.array([0]*a.landscape.length))
                            score = L.heights[np.argmax(results)]
                            scores.append(score)
                        a.score = np.mean(scores)
                    agents.sort(key=lambda x: x.score, reverse=True) # sort agents by expertise
                    expert = Team(agents[:per_team], L, trust_level=t)
                    
                    """Create diverse group"""
                    no_repeat = random.sample(pool, per_team)
                    repeat = [i for i in pool if i not in no_repeat]
                    d_heu = [[i]+random.sample(repeat, h_each-1) for i in no_repeat]
                    
                    diverse = Team([Agent(i+len(all_perm), d_heu[i], L, sigma) for i in range(per_team)], 
                                                    L, 
                                                    trust_level=t)
            
                    d_record = []
                    x_record = []

                    for i in starts:
                        np.random.seed()
                        d_record.append(diverse.tournament(i))
                        x_record.append(expert.tournament(i))
                   
                    df = df.append(pd.DataFrame([[s,
                                                  np.mean(d_record), 
                                                  np.mean(x_record), 
                                                  list(itertools.chain.from_iterable([a.h for a in diverse.members])), 
                                                  list(itertools.chain.from_iterable([a.h for a in expert.members])), 
                                                  t,
                                                  sigma,
                                                  poolsize
                                                 ]], columns=cols), ignore_index=True)
    return(df)

# In[86]:

cols = ['smoothness', 'diverse', 'expert', 
            'd_heuristics', 'x_heuristics', 
            'trust', 'sigma', 'poolsize']
data = pd.DataFrame(columns=cols)

all_results = []
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run) for i in range(2)]

        for f in concurrent.futures.as_completed(results):
            all_results.append(f.result())

data = data.append(pd.concat(all_results, ignore_index=True))

data.to_csv('test.csv', index=False)
