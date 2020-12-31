import sys
import numpy as np

gridfile=''
policyfile=''

for i in range(len(sys.argv)):
    if sys.argv[i]=='--grid':
        gridfile=sys.argv[i+1]
    elif sys.argv[i]=='--value_policy':
        policyfile=sys.argv[i+1]


grid=np.loadtxt(gridfile)
grid=np.array(grid,dtype=int)
temp=np.loadtxt(policyfile)
temp=np.array(temp,dtype=int)
policy=np.array(temp[:,1])
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
                start=(i,j)
            elif grid[i][j]==3:
                end=(i,j)


actions=np.zeros((m,n),dtype=int)

for i in range(m):
    for j in range(n):
        if states[i][j]==-1:
            actions[i][j]=-1
        else:
            actions[i][j]=policy[states[i][j]]


cur=start

path=[]

# print(cur,start,end)

while cur!=end:
    # print(cur)
    i,j=cur
    action=actions[i][j]
    if action==0:
        path.append('N')
        cur=(i-1,j)
    elif action==1:
        path.append('S')
        cur = (i+1, j)
    elif action==2:
        path.append('E')
        cur = (i, j+1)
    else:
        path.append('W')
        cur = (i, j-1)

print(" ".join(path))
