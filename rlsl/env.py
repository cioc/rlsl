import math
import numpy as np
import random
import skiplist
import torch
import torch.distributions as distributions 

class UniformGenerator(object):
    def __init__(self, keys):
        self.keys = keys
        self.dist = distributions.Uniform(torch.tensor([0.0]), torch.tensor([float(len(self.keys))]))
    
    def sample(self):
        v = int(self.dist.sample().item())
        return self.keys[v]

class PeakGenerator(object):
    def __init__(self, keys, peaks):
        self.keys = keys
        peak_height = .75 / peaks
        other_height = .25 / (len(self.keys) - peaks)
        arr = []
        for _ in range(len(self.keys)):
            arr.append(other_height)
        selected = []
        while len(selected) < peaks:
            k = random.randint(0, len(self.keys))
            if k in selected:
                continue
            selected.append(k)
            arr[k] = peak_height
        self.dist = distributions.Categorical(torch.tensor(arr))
        self.peaks = selected

    def sample(self):
        v = int(self.dist.sample().item())
        return self.keys[v]   

class SkipListWrapper(object):
    def __init__(self, device, h, kv_pairs):
        self.device = device
        self.h = h
        self.kv_pairs = kv_pairs

        self.structure_matrix = None
        self.reset() # build matrix

    def read(self, k):
        v = skiplist.find_node_sl(self.sl, k)
        return v.value

    def reset(self):
        self.structure_matrix = np.zeros((self.h, len(self.kv_pairs)), dtype=np.float32)
        for i in range(self.h):
            self.structure_matrix[i][0] = 1.0

        for i in range(len(self.kv_pairs)):
            self.structure_matrix[self.h - 1][i] = 1.0
        
        self.sl = skiplist.NewSL(self.kv_pairs, self.structure_matrix)
        self.write_head = random.randint(1, len(self.kv_pairs) - 1)
        
    def state(self):
        structure = np.copy(self.structure_matrix)
        structure /= 2.0

        position = np.zeros((self.h, len(self.kv_pairs)), dtype=np.float32)
        for i in range(self.h):
            position[i][self.write_head] = 0.5

        access_pattern = np.zeros((self.h, len(self.kv_pairs)), dtype=np.float32)
        accums = np.zeros((self.h, len(self.kv_pairs)), dtype=np.float32)
        accessed = np.zeros((self.h, len(self.kv_pairs)), dtype=np.float32)

        # create a key --> position map
        key_pos = {}

        curr = self.sl
        while curr.child:
            curr = curr.child
        pos = 0
        while curr.next:
            key_pos[curr.key] = pos
            pos += 1
            curr = curr.next
        key_pos[curr.key] = pos

        head_pointer = self.sl
        height = 0
        accum_total = 0.0
        accums_total = 0.0
        accessed_total = 0.0
        while head_pointer.child:
            curr = head_pointer
            while curr.next:
                access_pattern[height][key_pos[curr.key]] = curr.visited
                accum_total += curr.visited
                curr = curr.next
            access_pattern[height][key_pos[curr.key]] = curr.visited
            accum_total += curr.visited
            height += 1
            head_pointer = head_pointer.child
        curr = head_pointer
        while curr.next:
            access_pattern[height][key_pos[curr.key]] = curr.visited
            accums[height][key_pos[curr.key]] = curr.accum
            accessed[height][key_pos[curr.key]] = curr.accessed

            accum_total += curr.visited
            accums_total += curr.accum
            accessed_total += curr.accessed
            curr = curr.next
        access_pattern[height][key_pos[curr.key]] = curr.visited
        accums[height][key_pos[curr.key]] = curr.accum
        accessed[height][key_pos[curr.key]] = curr.accessed
        accum_total += curr.visited     
        accums_total += curr.accum
        accessed_total += curr.accessed

        access_pattern /= accum_total
        accums /= accums_total
        accessed /= accessed_total

        output = np.zeros((5, self.h,len(self.kv_pairs)), dtype=np.float32)
        output[0] = structure
        output[1] = access_pattern
        output[2] = position
        output[3] = accums
        output[4] = accessed

        tensor = torch.from_numpy(output)
        tensor = torch.unsqueeze(tensor, 0)

        res = tensor.to(self.device)

        return res
    def step(self, action):
        if action == 4:
            # go left
            self.write_head -= 1
            if self.write_head < 1:
                self.write_head = len(self.kv_pairs) - 1
        elif action == 5:
            # go right
             self.write_head += 1
             if self.write_head > (len(self.kv_pairs) - 1):
                 self.write_head = 1
        else:
            for i in range(0, self.h-1):
                self.structure_matrix[i][self.write_head] = 0

            for i in range(self.h - 2, self.h - 2 - action, -1):
                self.structure_matrix[i][self.write_head] = 1

        self.sl = skiplist.NewSL(self.kv_pairs, self.structure_matrix)

    def reset_counters(self):
        curr = self.sl
        while curr.child:
            curr = curr.child
        while curr.next:
            curr.accum = 0
            curr.visited = 0
            curr.accessed = 0
            curr = curr.next
        curr.accum = 0
        curr.visited = 0
        curr.accessed = 0

    def avg_access(self):
        avg_accesses = []

        curr = self.sl
        while curr.child:
            curr = curr.child
        while curr.next:
            if curr.accessed != 0:
                avg_accesses.append(float(curr.accum) / float(curr.accessed))
            else:
                avg_accesses.append(0.0)
            curr = curr.next
        if curr.accessed != 0:
            avg_accesses.append(float(curr.accum) / float(curr.accessed))   
        else:
            avg_accesses.append(0.0) 

        return np.mean(avg_accesses)

class Environment(object):
    def __init__(self, skip_list, keys):
        self.skip_list = skip_list
        self.traffic_generator = UniformGenerator(keys)
        self.sample_batch = 1000
        self.turn_count = 0
        self.max_turn = 2000
        self.keys = keys
        self.peaks = []

    def reset(self):
        #if random.random() > 0.0:
        #    print("Using Uniform Generator...")
        #    self.traffic_generator = UniformGenerator(self.keys)
        #    self.peaks = []
        #else:
        peak_count =  random.randint(1, 10)
        print("Using Peak Generator: %d..." % (peak_count))
        self.traffic_generator = PeakGenerator(self.keys, peak_count)
        self.peaks = self.traffic_generator.peaks
        self.skip_list.reset()
        self.turn_count = 0
        for _ in range(self.sample_batch):
            k = self.traffic_generator.sample()
            value = self.skip_list.read(k)
            if int(k) != value:
                raise Exception("Value did not match: %s %d" % (k, value))

    def state(self):
        return self.skip_list.state()

    def step(self, action):
        self.skip_list.step(action)
        self.skip_list.reset_counters()
        for _ in range(self.sample_batch):
            k = self.traffic_generator.sample()
            value = self.skip_list.read(k)
            if int(k) != value:
                raise Exception("Value did not match: %s %d" % (k, value))
        self.turn_count += 1
        return None, 1 / math.log(self.skip_list.avg_access()), self.turn_count > self.max_turn, None 
