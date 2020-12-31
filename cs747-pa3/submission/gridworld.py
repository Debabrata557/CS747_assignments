import numpy as np
import matplotlib.pyplot as plt
import sys


class WindyGrid():
    __sw = False
    __rows = 7
    __cols = 10
    __wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    __start = (3, 0)
    __end = (3, 7)
    __cur = __start
    __moves = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, 1),
        3: (0, -1),
        4: (-1, 1),
        5: (1, 1),
        6: (1, -1),
        7: (-1, -1)
    }

    def __init__(self, sw):
        self.__sw = sw

    def is_valid(self, x, y):
        return self.__rows > x >= 0 and self.__cols > y >= 0

    def decode(self, state):
        return state // self.__cols, state % self.__cols

    def encode(self, cell):
        # print(cell)
        return cell[0] * self.__cols + cell[1]

    def move(self, state, action):
        x, y = self.decode(state)
        next_x=x
        next_y=y
        # print((x,y),action)
        next_x += self.__moves[action][0]
        next_y += self.__moves[action][1]
        # print(y,state)
        next_x -=self.__wind[y]
        if self.__wind[y] > 0:
            if self.__sw:
                temp=[-1,0,1]
                c = np.random.choice(3)
                next_x-=temp[c]
        # print(self.decode(state),action,(x,y))
        next_x=max(0,next_x)
        next_x=min(self.__rows-1,next_x)
        next_y=max(0,next_y)
        next_y=min(self.__cols-1,next_y)
        # print((x,y))
        return self.encode((next_x, next_y)), -1
            # if (x,y)==self.__end:
            #     return self.encode((x,y)), -1
            # else:
            #     return self.encode((x,y)),-1

    def start_state(self):
        return self.encode(self.__start)
    def end_state(self):
        return self.encode(self.__end)
    def reset(self):
        self.__cur = self.__start
    def getdimension(self):
        return self.__rows,self.__cols

def sarsa(eps,alpha,mdp,gamma,num_actions,episodes):
    timesteps=[]
    rows,cols=mdp.getdimension()
    start=mdp.start_state()
    end=mdp.end_state()
    Q=np.zeros((rows*cols,num_actions))
    count=0
    # while count<8000:
    for episode in range(episodes):
        s=start
        a = np.argmax(Q[s])
        h = np.random.binomial(1, eps)
        if h:
            a = np.random.choice(num_actions)
        while s!=end:
            count += 1
            next_s, reward = mdp.move(state=s, action=a)
            a_dash=np.argmax(Q[next_s])
            h = np.random.binomial(1, eps)
            if h:
                a_dash = np.random.choice(num_actions)
            Q[s,a]=Q[s,a]+alpha*(reward+gamma*Q[next_s,a_dash]-Q[s,a])
            s=next_s
            a=a_dash
        episode+=1
        timesteps.append(count)
    return timesteps

def q_learning(eps,alpha,mdp,gamma,num_actions,episodes):
    timesteps = []
    rows, cols = mdp.getdimension()
    start = mdp.start_state()
    end = mdp.end_state()
    Q=np.zeros((rows*cols,num_actions))
    count = 0
    # while count < 8000:
    for episode in range(episodes):
        s=start
        while s!=end:
            count += 1
            a = np.argmax(Q[s])
            h = np.random.binomial(1, eps)
            if h:
                a = np.random.choice(num_actions)
            next_s, reward = mdp.move(state=s, action=a)
            Q[s,a]=Q[s,a]+alpha*(reward+gamma*np.max(Q[next_s])-Q[s,a])
            s=next_s
        timesteps.append(count)
    return timesteps

def expected_sarsa(eps,alpha,mdp,gamma,num_actions,episodes):
    timesteps = []
    rows, cols = mdp.getdimension()
    start = mdp.start_state()
    end = mdp.end_state()
    Q=np.zeros((rows*cols,num_actions))
    count = 0
    # while count <8000:
    for episode in range(episodes):
        s=start
        while s!=end:
            count += 1
            a = np.argmax(Q[s])
            h = np.random.binomial(1, eps)
            if h:
                a = np.random.choice(num_actions)
            next_s, reward = mdp.move(state=s, action=a)
            temp=np.zeros(num_actions)
            temp.fill(eps/num_actions)
            temp[np.argmax(Q[next_s])]+=1-eps
            Q[s,a]=Q[s,a]+alpha*(reward+gamma*np.dot(temp,Q[next_s])-Q[s,a])
            s=next_s
        timesteps.append(count)
    return timesteps


num_actions=4

eps=float(sys.argv[1])
alpha=float(sys.argv[2])
episodes=int(sys.argv[3])
stochasticity=int(sys.argv[4])
king_moves=int(sys.argv[5])
algos=sys.argv[6:]


if king_moves:
    num_actions=8

mdp=WindyGrid(sw=stochasticity)

plt.style.use('seaborn-darkgrid')

for algo in algos:
    timesteps = np.zeros(episodes)
    for seed in range(50):
        np.random.seed(seed)
        temp = np.zeros(episodes)
        if algo == 'q_learning':
            temp = np.array(q_learning(eps, alpha, mdp, 1, num_actions, episodes))
        if algo == 'sarsa':
            temp = np.array(sarsa(eps, alpha, mdp, 1, num_actions, episodes))
        if algo == 'e_sarsa':
            temp = np.array(expected_sarsa(eps, alpha, mdp, 1, num_actions, episodes))
        timesteps += temp

    timesteps /= 50
    plt.plot(timesteps,np.arange(1,episodes+1))

title=[]
if king_moves:
    title.append("King's move allowed")
else:
    title.append("King's move not allowed")

if stochasticity:
    title.append("Stochastic Wind")
else:
    title.append("No Stochastic Wind")

plt.legend(algos)
plt.title(" & ".join(title))
# plt.title("T5 : Comparison of Q learning, sarsa, expected sarsa agents with basic moves")
plt.ylabel('Episodes')
plt.xlabel('Time Steps')
plt.savefig('{}.png'.format(str(sys.argv[1:])))
# plt.show()








