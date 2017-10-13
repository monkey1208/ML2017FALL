import csv
import sys

def testPredict(w, x, b, threshold, error):
    tmp = 0
    for i in range(len(w)):
        tmp += w[i]*x[i]
    if tmp + b > threshold:
        return tmp + b + error
    return tmp + b

def main():
    w = []
    b = 0
    threshold = 130
    fixed_error = 25
    with open(sys.argv[1], 'r') as f:
        fst = True
        for row in csv.reader(f):
            if fst:
                fst = False
                continue
            if row[0] != 'b':
                w.append(float(row[1]))
            else:
                b = float(row[1])
    with open(sys.argv[2], 'r') as f:
        line = 1
        predict = []
        for row in csv.reader(f):
            if line % 18 == 10:
                ans = {'id': row[0], 'value': testPredict(w, [float(x) for x in row[4:]], b, threshold, fixed_error)}
                predict.append(ans)
            line += 1
    with open(sys.argv[3], 'w') as f:
        keys = predict[0].keys()
        o = csv.DictWriter(f, keys)
        o.writeheader()
        o.writerows(predict)
            
if __name__ == "__main__":
    main()
