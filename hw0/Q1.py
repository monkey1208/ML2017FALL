import sys
def main():
    index = 0
    d_index = dict()
    d_count = dict()
    fname = sys.argv[1]
    with open(fname, 'r') as f:
        line = f.readline().strip('\n')
        words = line.split(' ')
        for word in words:
            if word in d_index.keys():
                d_count[word] += 1
            else:
                d_index[word] = index
                index += 1
                d_count[word] = 1
    count = 1
    with open("Q1.txt", 'w') as f:
        for out in d_index.keys():
            f.write(out+" "+str(d_index[out])+" "+str(d_count[out]))
            if count < len(d_index):
                f.write("\n")
            count += 1
if __name__ == "__main__":
    main()
