import numpy as np
import re
import sys
import time
import pulp as pl



path=''
algo=''

for i in range(len(sys.argv)):
    if sys.argv[i]=='--mdp':
        path=sys.argv[i+1]
    elif sys.argv[i]=='--algorithm':
        algo=sys.argv[i+1]


numStates=0
numActions=0
start=0
end=[]
T=np.zeros((100,100,100))
R=np.zeros((100,100,100))
continuing=1
gamma=-1


f=open(path,'r')

for line in f:
    sent=line.strip('\n')
    words=re.split('\s+',sent)
    if words[0]=='numStates':
        numStates=int(words[1])
    elif words[0]=='numActions':
        numActions=int(words[1])
        T=np.resize(T,(numStates,numActions,numStates))
        R=np.resize(R,(numStates,numActions,numStates))
    elif words[0]=='start':
        start=int(words[1])
    elif words[0]=='end':
        end=list(map(int,words[1:]))
    elif words[0]=='transition':
        s1,a,s2,r,p=list(map(float,words[1:]))
        s1=int(s1)
        a=int(a)
        s2=int(s2)
        T[s1][a][s2]=p
        R[s1][a][s2]=r
    elif words[0]=='mdptype':
        if words[1]=='episodic':
            continuing=0
    elif words[0]=='discount':
        gamma=float(words[1])


f.close()

def value_iteration():
    v_prev=np.zeros(numStates)
    policy=np.zeros(numStates)
    while True:
        temp = np.einsum('ijk,ijk->ij', T, (R + (gamma * v_prev)))
        v_next = np.max(temp,axis=1)
        policy=np.argmax(temp,axis=1)
        if(np.max(np.abs(v_prev-v_next))<=1e-12):
            break
        v_prev=np.copy(v_next)
    return v_prev,policy

def policy_evaluation(pi):
    # print(pi)
    I=np.eye(numStates)
    temp = np.einsum('ij,ij->i', T[np.arange(numStates),pi,:], R[np.arange(numStates),pi,:])
    # print(temp)
    v=(np.linalg.inv(I-gamma*(T[np.arange(numStates),pi,:]))).dot(temp)
    return v

def policy_iteration():
    pi=np.random.choice(numActions,numStates)
    while True:
        v_pi = policy_evaluation(pi)
        can_be_improved = False
        # q_pi=np.zeros(numStates)
        # print(pi)
        for i in range(numStates):
            temp = np.einsum('ij,ij->i', T[i, :, :], (R[i, :, :] + (gamma * v_pi)))
            if np.max(temp) > v_pi[i] + 1e-12 and pi[i]!=np.argmax(temp):
                # print(np.max(temp),v_pi[i],i)
                can_be_improved = True
                pi[i] = np.argmax(temp)
        if can_be_improved==False:
            break
    return policy_evaluation(pi),pi

def LP():
    problem = pl.LpProblem('v*', pl.LpMinimize)
    v = np.empty(numStates, dtype=object)
    for i in range(numStates):
        v[i] = pl.LpVariable('v({})'.format(i))
    for s in range(numStates):
        for a in range(numActions):
            problem += (v[s] - (np.sum(T[s, a, :] * (R[s, a, :] + gamma * v)))) >= 0
    problem += np.sum(v)
    problem.solve(pl.PULP_CBC_CMD(msg=False))
    # print(pl.LpStatus[status])
    v_star=np.zeros(numStates)
    p_star=np.zeros(numStates)
    for i in range(numStates):
        v_star[i]=pl.value(v[i])
    for i in range(numStates):
        p_star[i]=np.argmax(np.einsum('ij,ij->i',T[i,:,:],R[i,:,:]+gamma*v_star))
    return v_star,p_star

start_t=time.time()

if algo=='vi':
    v_star,p_star=value_iteration()
    for v_i,p_i in zip(v_star,p_star):
        print(round(v_i,6),int(p_i))

if algo=='hpi':
    v_star, p_star = policy_iteration()
    for v_i, p_i in zip(v_star, p_star):
        print(round(v_i, 6), int(p_i))

if algo=='lp':
    v_star, p_star = LP()
    for v_i, p_i in zip(v_star, p_star):
        print(round(v_i, 6), int(p_i))

end_t=time.time()

# print(end_t-start_t)