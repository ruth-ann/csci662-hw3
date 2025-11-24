import argparse
import os
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", nargs='+', required=True, help="Prediction files (space-separated)")
    parser.add_argument("-t", type=str, required=True, help="Gold answers file")
    args = parser.parse_args()

    with open(args.t, "r") as f:
        truth = f.read().strip().split('\n')

    with open("datasets/questions.dev.txt", "r") as f:
        questions = f.read().strip().split('\n')

    assert all(len(open(f).read().strip().split('\n')) == len(truth) for f in args.p), "All prediction files must have the same number of lines as gold answers and questions"

    combined_file = os.path.join(os.path.dirname(args.p[0]), "analysis_all.csv")
    headers = ["question", "question_words", "question_chars"]
    model_names = []
    for f in args.p:
        name = os.path.splitext(os.path.basename(f))[0]
        model_names.append(name)
        headers.append(f"prediction_{name}")
        headers.append(f"correct_{name}")
    headers += ["models_correct", "num_models_correct"]

    with open(combined_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        preds_data = []
        for f in args.p:
            with open(f, "r") as pf:
                preds_data.append(pf.read().strip().split('\n'))

        for i, q in enumerate(questions):
            row = [q, len(q.split()), len(q)]
            correct_models = []

            for j, f in enumerate(args.p):
                pred = preds_data[j][i]
                gold = truth[i]

                matched = 0
                for gold_item in gold.split('\t'):
                    gold_item = gold_item.strip()
                    # strip off possible stray " 
                    gold_item = gold_item.replace('"', '')
                    if gold_item == '':
                        continue
                    if gold_item in pred:
                        matched = 1
                        correct_models.append(model_names[j])
                        break

                row.append(pred)
                row.append(matched)

            row.append(",".join(correct_models))
            row.append(len(correct_models))
            writer.writerow(row)

    print(f"Combined analysis saved to {combined_file}")
