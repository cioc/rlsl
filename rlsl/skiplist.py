class Node(object):
    def __init__(self):
        self.key = None
        self.value = None
        self.next = None
        self.child = None
        self.accum = 0
        self.visited = 0
        self.accessed = 0

def NewLL(kv_pairs):
    head = None
    curr = None
    kv_pairs = sorted(kv_pairs, key=lambda kv: kv[0])
    for k, v in kv_pairs:
        new = Node()
        new.key = k
        new.value = v
        if not head:
            head = new
            curr = new
        else:
            curr.next = new
            curr = new
    return head
    
def find_node(head, k, accum):
    curr = head
    while curr.next:
        curr.visited += 1
        if accum:
            accum += 1
        if curr.key == k:
            curr.accessed += 1
            if accum:
                curr.accum += accum                
            return curr
        curr = curr.next
    if curr.key == k and not curr.next:
        curr.visited += 1
        if accum:
            curr.accessed += 1
            curr.accum += accum + 1
        return curr
    return None


def NewSL(kv_pairs, structure):
    base = NewLL(kv_pairs)
    kv_pairs = sorted(kv_pairs, key=lambda kv: kv[0])
    for i in range(len(structure) - 2, -1, -1):
        line = structure[i]
        subset = []
        for k in range(len(line)):
            if line[k] == 1:
                subset.append(kv_pairs[k])
        layer = NewLL(subset)
        curr = layer
        curr.child = base
        while curr.next:
            curr.next.child = find_node(base, curr.next.key, None)
            curr = curr.next
        base = layer
    return base
        

def find_node_sl(sl, k):
    curr = sl
    accum = 0
    while curr.child:
        accum += 1
        while curr.next and curr.next.key <= k:
            accum += 1
            curr.accum += accum 
            curr.visited += 1
            curr = curr.next
        curr.visited += 1
        curr.accum += accum
        curr = curr.child
    return find_node(curr, k, accum)  

'''
matrix = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

sl = NewSL([("a",1), ("d",0),("b",2),("c",3),("e",3),("f",3),("g",3),("h",3),("i",3),("j",3),("k",3)], matrix)

for k in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]:
    value = find_node_sl(sl, k)
    print(value.key, value.value, value.accum, value.visited)
'''