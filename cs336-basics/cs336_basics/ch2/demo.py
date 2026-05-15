from multiprocessing import Queue, Process

from sortedcontainers import sorteddict


def worker(q):
    q.put("来自子进程的数据")


if __name__ == "__main__":
    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    print("主进程收到:", q.get())
    p.join()
    sd: sorteddict = {}
