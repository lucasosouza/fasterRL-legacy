## goal is to setup two processes running in parallel and exchanging information

from multiprocessing import Pool, Queue
from multiprocessing import Process, Manager, Value, Array
from time import sleep
import random

EXPERIENCE_BUFFER_SIZE = 50 # 100000
TRANSFER_BUFFER_SIZE = 10


# one method does not see the array manipulated in the other process
# that won't work 
# has to be a dataset they share
# go back to manager

def method(agent_id, give, receive, req_give, req_receive):

    experiences = []
    while len(experiences) < EXPERIENCE_BUFFER_SIZE:

        # live a few experiences
        for _ in range(10):
            sleep(0.1)
            experiences.append(random.random())

        # give
        req_give_threshold = req_give.get()
        filtered_experiences = list(filter(lambda x: x < req_give_threshold, experiences))
        if len(filtered_experiences) > 0:
            experiences_to_share = [random.choice(filtered_experiences) for _ in range(TRANSFER_BUFFER_SIZE)]
            # put all experiences in queue
            for exp in experiences_to_share:
                give.put(exp)
        print("Agent {} - sharing new experiences: {:d}".format(agent_id, len(give)))

        # receive
        print("Agent {} - receiving new experiences: {:d}".format(agent_id, len(receive)))
        # get all experiences from queue
        for i in range(len(receive)):
            experiences.append(receive.get())
        del receive[:]

        # request more
        req_receive.put(random.random())

        # print("Agent {} - issuing a new request: {:.2f}".format(agent_id, request[0]))
        print(agent_id)
        print("give", give)
        print("receive", receive)


    print("Done process {}".format(agent_id))
    print(experiences)

if __name__ == "__main__":


    # vecs doesnt have to be a parallel dataset. only the lists which are part of it
    vecs = []

    with Manager() as manager:
        
        with Pool(processes=2) as pool: 

            processes = []

            vec_share = []
            vec_request = []
            for idx in range(2):
                vec_share.append(manager.Queue())
                vec_request.append(manager.Queue())

            for idx in range(2):
                # # vecs.append((manager.list(), manager.list(), Value('d', 0.0)))                             
                # vecs.append(([], [], [0.0])) # trying with regular lists
                give = vec_share[abs(0-idx)]
                receive = vec_share[abs(1-idx)]
                req_give = vec_request[abs(0-idx)]
                req_receive = vec_request[abs(1-idx)]
                p = pool.apply_async(method, [idx, give, receive, req_give, req_receive])
                # when 0, first is position 0, second is 1
                # when 1, first is position 1, second is 0
                processes.append(p)

            for r in processes:
                r.get()




# Synchronized objects should only be shared between processes through inheritance
# can't share sync objects between processes
# maybe I can share 

## basic setup for parallelization. now I need to include data exchanging methods
## using data exchange


"""

from multiprocessing import Pool
from multiprocessing import Process, Manager, Value, Array
from time import sleep

def method(process_num, vec):
    vec.append(process_num)
    sleep(2-process_num/5)
    print("Done process {}".format(process_num))

if __name__ == "__main__":

    with Manager() as manager:

        vec = manager.list()

        with Pool(processes=2) as pool: 

            processes = []

            for i in range(10):
                p = pool.apply_async(method, [i, vec])
                processes.append(p)

            for r in processes:
                r.get()

            print(vec)

"""