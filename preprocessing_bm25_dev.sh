#!/bin/bash
python3 generator.py -r bm25 -n bm25_mindf3_index.pkl -k 1 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_bm25_mindf3_index_k1_dev.answers.txt

python3 generator.py -r bm25 -n bm25_k05_index.pkl -k 1 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_bm25_k05_index_k1.answers_dev.txt

python3 generator.py -r bm25 -n bm25_b0_index.pkl -k 1 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_bm25_bm25_b0_index_index_k1_dev.answers.txt

