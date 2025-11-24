import argparse

# Fixed questions file
QUESTIONS_FILE = "datasets/questions.dev.txt"

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", nargs='+', required=True, help="Prediction files")
    parser.add_argument("-t", type=str, required=True, help="Gold answers file")
    args = parser.parse_args()

    # Load questions and gold answers
    with open(QUESTIONS_FILE, "r") as f:
        questions = [line.strip() for line in f]

    with open(args.t, "r") as f:
        gold_answers = [line.strip() for line in f]

    # Prepare categories
    categories = {
        "who": [i for i, q in enumerate(questions) if "who" in q.lower()],
        "where": [i for i, q in enumerate(questions) if "where" in q.lower()],
        "when": [i for i, q in enumerate(questions) if "when" in q.lower()],
        "all_others": [i for i, q in enumerate(questions) if all(w not in q.lower() for w in ["who","where","when"])]
    }

    # Store per-model accuracies
    all_accuracies = {cat: [] for cat in categories}

    for model_file in args.p:
        with open(model_file, "r") as f:
            preds = [line.strip() for line in f]

        for cat, idxs in categories.items():
            correct = sum(int(any(ans.lower() in preds[i].lower() for ans in gold_answers[i].split('\t'))) for i in idxs)
            acc = correct / len(idxs) if idxs else 0
            all_accuracies[cat].append(acc)
            print(f"Model: {model_file}  | Category: {cat} | Accuracy: {acc:.3f} ({correct}/{len(idxs)})")

    # Compute averages across all models
    print("\n=== Average Accuracy Across All Models ===")
    for cat, acc_list in all_accuracies.items():
        avg_acc = sum(acc_list)/len(acc_list) if acc_list else 0
        print(f"{cat}: {avg_acc:.3f}")
