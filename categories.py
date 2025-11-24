import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", nargs='+', required=True)
    parser.add_argument("-t", type=str, required=True)
    args = parser.parse_args()

    with open("datasets/questions.dev.txt", "r") as question_file:
        questions = [line.strip() for line in question_file]

    with open(args.t, "r") as gold_answer_file:
        gold_answers = [line.strip() for line in gold_answer_file]

    categories = {
        "who": [index for index, question in enumerate(questions) if "who" in question.lower()],
        "where": [index for index, question in enumerate(questions) if "where" in question.lower()],
        "when": [index for index, question in enumerate(questions) if "when" in question.lower()],
        "others": [
            index for index, question in enumerate(questions)
            if all(keyword not in question.lower() for keyword in ["who", "where", "when"])
        ]
    }

    all_accuracies = {category_name: [] for category_name in categories}

    for model_output_file in args.p:
        with open(model_output_file, "r") as prediction_file:
            predictions = [line.strip() for line in prediction_file]

        for category_name, index_list in categories.items():
            correct_count = 0

            for index in index_list:
                possible_answers = gold_answers[index].split("\t")
                if any(answer.lower() in predictions[index].lower() for answer in possible_answers):
                    correct_count += 1

            accuracy = correct_count / len(index_list) if index_list else 0
            all_accuracies[category_name].append(accuracy)

    for category_name, accuracy_list in all_accuracies.items():
        average_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
        print(f"{category_name}: {average_accuracy:.3f}")
