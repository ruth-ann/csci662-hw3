import pandas as pd
import matplotlib.pyplot as plt
import os

analysis_file = "analysis_all.csv"
df = pd.read_csv(analysis_file)
model_cols = [c for c in df.columns if c.startswith("correct_")]

counts = df[model_cols].sum()
counts.index = [col.replace("correct_", "") for col in counts.index]

plt.figure(figsize=(10,6))
counts.plot(kind="bar", color="skyblue")
plt.title("Number of times each model is correct")
plt.xlabel("Model")
plt.ylabel("Count Correct")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
bar_chart_file = os.path.join(os.path.dirname(analysis_file), "model_correct_count.png")
plt.savefig(bar_chart_file)
plt.close()
print(f"Saved bar chart to {bar_chart_file}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
df.boxplot(column="question_words", by="num_models_correct", grid=True)
plt.title("Question Words")
plt.suptitle("")
plt.xlabel("Number of Models Correct")
plt.ylabel("Words")
plt.subplot(1,2,2)
df.boxplot(column="question_chars", by="num_models_correct", grid=True)
plt.title("Question Characters")
plt.suptitle("")
plt.xlabel("Number of Models Correct")
plt.ylabel("Chars")
plt.tight_layout()
scatter_file = os.path.join(os.path.dirname(analysis_file), "num_models_correct_boxplots.png")
plt.savefig(scatter_file)
plt.close()
print(f"Saved boxplots to {scatter_file}")
