import sys
import numpy as np
path=sys.argv[2]

grid=np.loadtxt(path,dtype=int)
# print(grid)


numActions=4
# print(numStates,numActions)

states=np.zeros(grid.shape,dtype=int)
start=-1
end=-1

m,n=grid.shape

cur=0
for i in range(m):
    for j in range(n):
        if grid[i][j]==1:
            states[i][j]=-1
        else:
            states[i][j]=cur
            cur+=1
            if grid[i][j]==2:
                start=states[i][j]
            elif grid[i][j]==3:
                end=states[i][j]

numStates=cur

print("numStates {}".format(numStates))
print("numActions {}".format(numActions))
print("start {}".format(start))
print("end {}".format(end))

for i in range(m):
    for j in range(n):
        if(grid[i][j]==3 or grid[i][j]==1):
            continue
        if i-1 >=0 and grid[i-1][j]!=1:
            print("transition {} {} {} {} {}".format(states[i][j], 0, states[i - 1][j], -1, 1))
        else:
            print("transition {} {} {} {} {}".format(states[i][j], 0, states[i][j], -10, 1))
        if i+1 < m and grid[i+1][j]!=1:
            print("transition {} {} {} {} {}".format(states[i][j], 1, states[i + 1][j], -1, 1))
        else:
            print("transition {} {} {} {} {}".format(states[i][j], 1, states[i][j], -10, 1))
        if j-1 >=0 and grid[i][j-1]!=1:
            print("transition {} {} {} {} {}".format(states[i][j], 3, states[i][j - 1], -1, 1))
        else:
            print("transition {} {} {} {} {}".format(states[i][j], 3, states[i][j], -10, 1))
        if j+1 < n and grid[i][j+1]!=1:
            print("transition {} {} {} {} {}".format(states[i][j], 2, states[i][j + 1], -1, 1))
        else:
            print("transition {} {} {} {} {}".format(states[i][j],2, states[i][j], -10, 1))



print("mdptype episodic")
print("discount 1")

