import argparse
import os
import pandas as pd

def detect_method(filename):
    filename_lower = filename.lower()
    if "dense" in filename_lower:
        return "dense"
    elif "bm25" in filename_lower:
        return "bm25"
    elif "tfidf" in filename_lower:
        return "tfidf"
    else:
        return None

def parse_retrieval_file(filepath):
    retrieval_values = []
    with open(filepath, "r") as file:
        for line in file:
            cleaned_line = line.strip().lower()
            if cleaned_line == "true":
                retrieval_values.append(1)
            elif cleaned_line == "false":
                retrieval_values.append(0)
            else:
                retrieval_values.append(None)
    return retrieval_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", nargs='+', required=True)
    parser.add_argument("-t", type=str, required=True)
    parser.add_argument("-r", nargs='*', default=[])
    args = parser.parse_args()

    with open(args.t, "r") as gold_answer_file:
        gold_answers = [line.strip() for line in gold_answer_file]

    model_predictions = {}
    for prediction_file_path in args.p:
        with open(prediction_file_path, "r") as prediction_file:
            model_predictions[prediction_file_path] = [line.strip() for line in prediction_file]

    retrieval_data = {}
    for retrieval_file_path in args.r:
        method_name = detect_method(retrieval_file_path)
        if method_name:
            retrieval_data[method_name] = parse_retrieval_file(retrieval_file_path)

    number_of_rows = len(gold_answers)
    valid_rows = []
    for row_index in range(number_of_rows):
        if all(retrieval_data[method_name][row_index] is not None for method_name in retrieval_data):
            valid_rows.append(row_index)

    summary = {}
    for model_file, prediction_list in model_predictions.items():
        method_name = detect_method(model_file)
        if method_name is None:
            continue

        retrieval_list = retrieval_data[method_name]

        counts = {
            "retrieval=true,correct": 0,
            "retrieval=true,wrong": 0,
            "retrieval=false,correct": 0,
            "retrieval=false,wrong": 0
        }

        df_rows = []

        for row_index in valid_rows:
            prediction_text = prediction_list[row_index].strip()
            gold_answer_list = [
                answer.strip().replace('"', '')
                for answer in gold_answers[row_index].split('\t')
                if answer.strip()
            ]

            correct_flag = int(any(answer in prediction_text for answer in gold_answer_list))
            retrieval_flag = retrieval_list[row_index]

            if retrieval_flag == 1 and correct_flag == 1:
                counts["retrieval=true,correct"] += 1
            elif retrieval_flag == 1 and correct_flag == 0:
                counts["retrieval=true,wrong"] += 1
            elif retrieval_flag == 0 and correct_flag == 1:
                counts["retrieval=false,correct"] += 1
            elif retrieval_flag == 0 and correct_flag == 0:
                counts["retrieval=false,wrong"] += 1

        summary[model_file] = {
            "method": method_name,
            "retrieval=true,correct": counts["retrieval=true,correct"],
            "retrieval=true,wrong": counts["retrieval=true,wrong"],
            "retrieval=false,correct": counts["retrieval=false,correct"],
            "retrieval=false,wrong": counts["retrieval=false,wrong"]
        }

    df_rows = []

    for row_index in valid_rows:
        row = {
            "row": row_index,
            "gold": gold_answers[row_index]
        }

        for model_file, prediction_list in model_predictions.items():
            row[model_file] = prediction_list[row_index]

        for method_name, retrieval_list in retrieval_data.items():
            row[f"retrieval_{method_name}"] = retrieval_list[row_index]

        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    combined_file = os.path.join(os.path.dirname(args.p[0]), "analysis_all.csv")
    df.to_csv(combined_file, index=False)

    for model_file, stats in summary.items():
        print(f"Model: {model_file}  (method: {stats['method']})")
        print(f"retrieval=true,correct = {stats['retrieval=true,correct']}")
        print(f"retrieval=true,wrong   = {stats['retrieval=true,wrong']}")
        print(f"retrieval=false,correct = {stats['retrieval=false,correct']}")
        print(f"retrieval=false,wrong   = {stats['retrieval=false,wrong']}\n")
