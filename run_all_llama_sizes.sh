#!/bin/bash


python3 generator.py -r bm25 -n bm25_index.pkl -k 1 -p ollama -m llama3.2:1b -i datasets/questions.dev.txt -o llama321_bm25_k1_dev.answers.txt

python3 generator.py -r bm25 -n bm25_index.pkl -k 1 -p ollama -m llama3.2:3b -i datasets/questions.dev.txt -o llama323_bm25_k1_dev.answers.txt

python3 generator.py -r bm25 -n bm25_index.pkl -k 1 -p ollama -m llama3.2:1b -i datasets/questions.test.txt -o llama321_bm25_k1_test.answers.txt

python3 generator.py -r bm25 -n bm25_index.pkl -k 1 -p ollama -m llama3.2:3b -i datasets/questions.test.txt -o llama323_bm25_k1_test.answers.txt
