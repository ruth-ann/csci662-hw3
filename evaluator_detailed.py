import argparse
import os
import pandas as pd

def detect_method(filename):
    fname = filename.lower()
    if "dense" in fname:
        return "dense"
    elif "bm25" in fname:
        return "bm25"
    elif "tfidf" in fname:
        return "tfidf"
    else:
        return None

def parse_retrieval_file(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().lower()
            if line == "true":
                data.append(1)
            elif line == "false":
                data.append(0)
            else:
                data.append(None)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", nargs='+', required=True, help="Prediction files")
    parser.add_argument("-t", type=str, required=True, help="Gold answers file")
    parser.add_argument("-r", nargs='*', default=[], help="Retrieval files")
    args = parser.parse_args()

    # Load gold answers
    with open(args.t, "r") as f:
        gold_answers = [line.strip() for line in f]

    # Load predictions
    model_preds = {}
    for f in args.p:
        with open(f, "r") as pf:
            model_preds[f] = [line.strip() for line in pf]

    # Load retrieval files
    retrieval_data = {}
    for rfile in args.r:
        method = detect_method(rfile)
        if method:
            retrieval_data[method] = parse_retrieval_file(rfile)

    # Remove rows with any None in retrieval
    n_rows = len(gold_answers)
    valid_rows = []
    for i in range(n_rows):
        if all(retrieval_data[m][i] is not None for m in retrieval_data):
            valid_rows.append(i)

    # Compute per-model stats using substring matching like simple scorer
    summary = {}
    for model_file, preds in model_preds.items():
        method = detect_method(model_file)
        if method is None:
            continue
        retriever = retrieval_data[method]
        counts = {"retrieval=true & correct":0, "retrieval=true & wrong":0,
                  "retrieval=false & correct":0, "retrieval=false & wrong":0}

        for i in valid_rows:
            pred = preds[i].strip()
            gold = [g.strip().replace('"','') for g in gold_answers[i].split('\t') if g.strip()]
            correct = int(any(g in pred for g in gold))
            rval = retriever[i]

            if rval == 1 and correct:
                counts["retrieval=true & correct"] += 1
            elif rval == 1 and not correct:
                counts["retrieval=true & wrong"] += 1
            elif rval == 0 and correct:
                counts["retrieval=false & correct"] += 1
            elif rval == 0 and not correct:
                counts["retrieval=false & wrong"] += 1

        summary[model_file] = {"method": method, **counts}

    # Save combined CSV
    df_rows = []
    for i in valid_rows:
        row = {"row": i, "gold": gold_answers[i]}
        for model_file, preds in model_preds.items():
            row[model_file] = preds[i]
        for method, rvals in retrieval_data.items():
            row[f"retrieval_{method}"] = rvals[i]
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    combined_file = os.path.join(os.path.dirname(args.p[0]), "analysis_all.csv")
    df.to_csv(combined_file, index=False)
    print(f"Saved updated analysis to {combined_file}\n")

    # Print summary
    print("=== Retriever Summary Per Model ===\n")
    for model_file, stats in summary.items():
        print(f"Model: {model_file}  (method: {stats['method']})")
        print(f"  retrieval=true  & correct = {stats['retrieval=true & correct']}")
        print(f"  retrieval=true  & wrong   = {stats['retrieval=true & wrong']}")
        print(f"  retrieval=false & correct = {stats['retrieval=false & correct']}")
        print(f"  retrieval=false & wrong   = {stats['retrieval=false & wrong']}\n")
