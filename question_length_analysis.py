import pandas as pd

analysis_file = "analysis_all.csv"
df = pd.read_csv(analysis_file)
model_cols = [c for c in df.columns if c.startswith("correct_")]

results = []

for col in model_cols:
    model_name = col.replace("correct_", "")
    correct_mask = df[col] == 1
    wrong_mask = df[col] == 0

    avg_words_correct = df.loc[correct_mask, "question_words"].mean()
    avg_chars_correct = df.loc[correct_mask, "question_chars"].mean()
    avg_words_wrong = df.loc[wrong_mask, "question_words"].mean()
    avg_chars_wrong = df.loc[wrong_mask, "question_chars"].mean()

    results.append({
        "model": model_name,
        "avg_words_correct": avg_words_correct,
        "avg_words_wrong": avg_words_wrong,
        "avg_chars_correct": avg_chars_correct,
        "avg_chars_wrong": avg_chars_wrong
    })

summary_df = pd.DataFrame(results)
print(summary_df)
summary_df.to_csv("question_length_by_model_correctness.csv", index=False)
print("Saved summary to question_length_by_model_correctness.csv")
