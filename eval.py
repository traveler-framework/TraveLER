import argparse
import csv
import glob
import os

def main(args):
    correct = 0
    total = 0
    
    path = os.path.join(os.getcwd(), "results", args.exp)
    os.chdir(path)
    for file in glob.glob('*.tsv'):
        with open(os.path.join(path, file)) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                total += 1
                row = [x.strip() for x in row]
                true_ans = row[4]
                pred_ans = row[-1]
                if true_ans == pred_ans:
                    correct += 1

    print(args.exp)
    print(f"correct: {correct}")
    print(f"total: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()

    main(args)
