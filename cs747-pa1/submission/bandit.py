import numpy as np
import sys
import time

import random


class BANDIT():
    def __init__(self,instance):
        self.num_of_arms=len(instance)
        self.__instance=instance
    def pull(self,num):
        # return np.random.binomial(1,self.__instance[num])
        return random.random()<self.__instance[num]


def initialize(bandit):
    pulls=np.array([bandit.pull(arm) for arm in range(bandit.num_of_arms)])
    return np.mean(pulls,axis=1)

def epsilon_greedy(epsilon,bandit,T):
    emp_means=np.zeros(bandit.num_of_arms)
    times_pulled=np.zeros(bandit.num_of_arms)
    cum_reward=0
    for t in range(1,T+1):
        explore=np.random.binomial(1,epsilon)
        arm = np.argmax(emp_means)
        if explore:
            arm=np.random.choice(bandit.num_of_arms)
        reward=bandit.pull(arm)
        times_pulled[arm] += 1
        emp_means[arm] = emp_means[arm] + (reward - emp_means[arm]) / times_pulled[arm]
        cum_reward+=reward
    return cum_reward

def kl(x,y):
    return x*np.log(x/y)+(1-x)*np.log((1-x)/(1-y))

# def find_best_q(u_a,p_a,t):
#     lo=p_a
#     hi=1
#     eps=1e-6
#     iter=0
#     while lo+eps<hi :
#         mid=(lo+hi)/2
#         if(u_a*kl(p_a,mid)<=np.log(t)+3*np.log(np.log(t))):
#             lo=mid
#         else:
#             hi=mid
#         # iter+=1
#     return lo

def find_best_q(times_pulled,emp_mean,t): # vectorized Implementation
    n=emp_mean.shape[0]
    lo=np.copy(emp_mean)
    hi=np.ones(n)
    hi-=1e-12
    eps=1e-3
    temp=np.arange(n)
    while np.sum(lo+eps < hi) != 0:
        extract=(lo+eps<hi)
        valid_indices=temp[extract]
        mid=(lo+hi)*0.5
        lt=(times_pulled[extract]*kl(emp_mean[extract],mid[extract])<=np.log(t)+3*np.log(np.log(t)))
        lo[valid_indices[lt]]=mid[valid_indices[lt]]
        hi[valid_indices[lt^True]]=mid[valid_indices[lt^True]]
    return lo



def ucb(bandit,T):
    emp_means = np.array([bandit.pull(arm) for arm in range(bandit.num_of_arms)],dtype=float)
    times_pulled = np.ones(bandit.num_of_arms)
    cum_reward = np.sum(emp_means)
    for t in range(bandit.num_of_arms+1,T+1):
        ucb=emp_means+np.sqrt(2*np.log(t)*(1/times_pulled))
        arm=np.argmax(ucb)
        reward=bandit.pull(arm)
        times_pulled[arm] += 1
        emp_means[arm] = emp_means[arm] + (reward - emp_means[arm]) / times_pulled[arm]
        cum_reward+=reward
    return cum_reward

def kl_ucb(bandit,T):
    emp_means = np.array([bandit.pull(arm) for arm in range(bandit.num_of_arms)], dtype=float)
    times_pulled = np.ones(bandit.num_of_arms)
    cum_reward = np.sum(emp_means)
    emp_means[emp_means==0]+=1e-12
    emp_means[emp_means==1]-=1e-12
    for t in range(bandit.num_of_arms+1,T+1):
        # klucb=list(map(find_best_q,times_pulled,emp_means,np.array([t]*bandit.num_of_arms,dtype=float)))
        klucb=find_best_q(times_pulled, emp_means, t)
        arm = np.argmax(klucb)
        reward=bandit.pull(arm)
        times_pulled[arm] += 1
        emp_means[arm] = emp_means[arm] + (reward - emp_means[arm]) / times_pulled[arm]
        cum_reward+=reward

    return cum_reward

def thompson_sampling(bandit,T):
    success=np.zeros(bandit.num_of_arms)
    failure=np.zeros(bandit.num_of_arms)
    cum_reward=0
    for t in range(1,T+1):
        x=np.array([np.random.beta(s+1,f+1) for s,f in zip(success,failure)])
        arm=np.argmax(x)
        reward=bandit.pull(arm)
        success[arm]+=reward
        failure[arm]+=1-reward
        cum_reward+=reward
    # print(success+failure)
    return cum_reward

def thompson_sampling_with_hint(bandit,T,perm_means):
    perm_means=np.sort(perm_means)
    pmfs=np.empty((bandit.num_of_arms,bandit.num_of_arms),dtype=float)
    pmfs.fill(1/bandit.num_of_arms)
    times_pulled=np.zeros(bandit.num_of_arms)
    cum_reward=0
    arms=np.arange(bandit.num_of_arms)

    for t in range(1,T+1):
        if np.max(pmfs[:,bandit.num_of_arms-1])<0.99:
            x = [np.random.choice(perm_means, p=pmfs[i, :]) for i in arms]
            arm = arms[np.random.choice(np.flatnonzero(x == np.max(x)))]
        else:
            arm=np.argmax(pmfs[:,bandit.num_of_arms-1])
        reward=bandit.pull(arm)
        temp=pmfs[arm,:]
        if reward:
            temp=temp*perm_means
        else:
            temp=temp*(1-perm_means)
        temp/=np.sum(temp)
        pmfs[arm,:]=temp
        cum_reward+=reward
        times_pulled[arm]+=1
    return cum_reward


path=''
algo=''
rs=-1
epsilon=-1
T=-1
for i in range(1,len(sys.argv)):
    temp=sys.argv[i]
    if temp=='--instance':
        path=sys.argv[i+1]
    elif temp=='--algorithm':
        algo=sys.argv[i+1]
    elif temp=='--randomSeed':
        rs=int(sys.argv[i+1])
    elif temp=='--epsilon':
        epsilon=float(sys.argv[i+1])
    elif temp=='--horizon':
        T=int(sys.argv[i+1])

f=open(path,'r')
instance=list(map(float,[line.rstrip('\n') for line in f]))
f.close()
np.random.seed(rs)
random.seed(rs)

bandit=BANDIT(instance)

start=time.time()
if algo=='epsilon-greedy':
    reward=epsilon_greedy(epsilon,bandit,T)
if algo=='ucb':
    reward=ucb(bandit,T)
if algo=='kl-ucb':
    reward=kl_ucb(bandit,T)
if algo=='thompson-sampling':
    reward=thompson_sampling(bandit,T)
if algo=='thompson-sampling-with-hint':
    reward=thompson_sampling_with_hint(bandit,T,np.sort(instance))

end=time.time()

regret = max(instance) * T - reward
print('{}, {}, {}, {}, {}, {}'.format(path, algo, rs, epsilon, T, round(regret, 3)))
# print(end-start)




