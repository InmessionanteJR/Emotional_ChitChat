import os,sys

def loaddict(file):
    f = open(file, "r")
    d = {}
    for line in f:
        line = line.strip("\n")
        #print("line", line)
        slots = line.split("\t")
        #print("slots:", slots)
        d[slots[0]] = slots[1]
    return d

def filter(str, dict):
    tokens = str.split()
    new_tokens = []
    for t in tokens:
        if t in dict:
            new_tokens.append(dict[t])
    return " ".join(new_tokens)

dict = loaddict(sys.argv[1])
fqo = open(sys.argv[2], "w")
fro = open(sys.argv[3], "w")
for line in sys.stdin:
    line = line.strip()
    q, r = line.split("\t")
    new_q = filter(q, dict)
    new_r = filter(r, dict)
    fqo.write(new_q + "\n")
    fro.write(new_r + "\n")


