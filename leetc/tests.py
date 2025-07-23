import time

N = 1000
cach = [[0]*N for _ in range(N)]
cache = {}

# Test list access
start = time.time()
for i in range(N):
    for j in range(N):
        cach[i][j] = i + j
end = time.time()
print("List access:", end - start)

# Test dict access
start = time.time()
for i in range(N):
    for j in range(N):
        cache[(i, j)] = i + j
end = time.time()
print("Dict access:", end - start)
