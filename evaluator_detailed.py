import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, required=True)
    parser.add_argument("-t", type=str, required=True)
    args = parser.parse_args()

    with open(args.p, "r") as f:
        to_score = f.read().strip().split('\n')

    with open(args.t, "r") as f:
        truth = f.read().strip().split('\n')

    with open("datasets/questions.dev.txt", "r") as f:
        questions = f.read().strip().split('\n')

    assert len(to_score) == len(truth) == len(questions), "Predictions, truth, and questions must have the same number of lines"

    correct = 0
    tsv_file = os.path.join(os.path.dirname(args.p), f"analysis_{os.path.basename(args.p)}.tsv")
    with open(tsv_file, "w") as out:
        out.write("Question\tPrediction\tGold\tCorrect\n")
        for q, pred, gold in zip(questions, to_score, truth):
            matched = False
            for gold_item in gold.split('\t'):
                gold_item = gold_item.strip()
                # strip off possible stray " 
                gold_item = gold_item.replace('"', '')
                if gold_item == '':
                    continue
                if gold_item in pred:
                    matched = True
                    break

            if matched:
                correct += 1

            out.write(f"{q}\t{pred}\t{gold}\t{matched}\n")

    print(f"{correct / len(to_score)}")
    print(f"Analysis saved to {tsv_file}")
