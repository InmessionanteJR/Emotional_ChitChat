import os,sys,chardet

is_rm_empty = True
print_opt = 1 # 0: print to stdout (with id) 1: print to stdout (without id) 2: write to file (src/tgt)
output_encoding = 'utf-8' # gb18030

def is_letter(s):
    if 'A' <= s <= 'Z' or 'a' <= s <= 'z':
        return True
    return False

def is_digit(s):
    if '0' <= s <= '9':
        return True
    return False

def is_letter_or_digit(s):
    return is_letter(s) or is_digit(s)

def is_empty(s):
    if ' ' == s:
        return True
    return False

def tokenize(s):
    ls = len(s)
    i, last_i = 0, 0
    res = []

    while i < ls:
        chinese_ch = True
        buf = ""
        j = 0
        #print("s:",s)
        #print("i:", i, "j:", j, "s[i]:", s[i])
        while (i + j < ls) and (is_letter_or_digit(s[i + j]) or (is_empty(s[i + j]) and is_letter_or_digit(s[i]))):
            j += 1
            chinese_ch = False
        if not chinese_ch:
            res += list(s[last_i:i]) # Chinese
            res += [s[i:i+j]] # English
            i += j
            last_i = i
        if chinese_ch:
            i += 1

    res += list(s[last_i:ls])

    if is_rm_empty:
        new_res = []
        for r in res:
            if r == " ":
                continue
            new_res.append(r)
        res = new_res[:]
    return res 

if print_opt == 2:
    out_src_file = sys.argv[1]
    out_tgt_file = sys.argv[2]
    fos = open(out_src_file, "w")
    fot = open(out_tgt_file, "w")

for line in sys.stdin:
    line = line.strip("\n").decode('utf-8')
    #line = line.strip("\n")
    slots = line.split("\t")
    # print('---------------------------')
    # print(slots)
    # print('---------------------------')
    # q, r = slots[7], slots[3]
    q, r = slots[0], slots[1]
    qt = tokenize(q)
    rt = tokenize(r)
    if len(qt) * len(rt) == 0:
        print("Error_Line_!!!")
        continue

    if print_opt == 2:
        res = " ".join(qt) + "\n"
        res = res.encode(output_encoding)
        fos.write(str(res))
        res = " ".join(rt) + "\n"
        res = res.encode(output_encoding)
        fot.write(str(res))

    if print_opt == 0:
        slots[7] = " ".join(qt)
        slots[3] = " ".join(rt)
        res = "\t".join(slots)
    elif print_opt == 1:
        res = " ".join(qt) + "\t" + " ".join(rt)

    if print_opt in [0, 1]:
        res = res.encode(output_encoding)
        #res = res.encode('utf-8')
        #print("res:", chardet.detect(res))
        print(res)

if print_opt == 2:
    fos.close()
    fot.close()
