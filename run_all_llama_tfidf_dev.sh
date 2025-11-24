#!/bin/bash

#gemma dev
python3 generator.py -r tfidf -n tfidf_index.pkl -k 3 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_tfidf_k3_dev.answers.txt

python3 generator.py -r tfidf -n tfidf_index.pkl -k 5 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_tfidf_k5_dev.answers.txt

python3 generator.py -r tfidf -n tfidf_index.pkl -k 7 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_tfidf_k7_dev.answers.txt

python3 generator.py -r tfidf -n tfidf_index.pkl -k 10 -p ollama -m llama3.1:8b -i datasets/questions.dev.txt -o llama318_tfidf_k10_dev.answers.txt