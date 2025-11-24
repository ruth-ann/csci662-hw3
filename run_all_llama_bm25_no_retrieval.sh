#!/bin/bash


python3 generator.py -r bm25 -n bm25_index.pkl -k 1 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_bm25_k1_vanilla_dev.answers.txt

python3 generator.py -r bm25 -n bm25_index.pkl -k 1 -p ollama -m llama3.1:8b -i datasets/questions.test.txt -o llama318_bm25_k1_vanilla_test.answers.txt
