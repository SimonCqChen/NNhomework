import heapq
import queue

q = queue.Queue(maxsize=5)
for i in range(10):
    q.put(i)
    print(q.qsize())
