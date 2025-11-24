#!/bin/bash

#gemma test

python3 generator.py -r tfidf -n tfidf_index.pkl -k 3 -p ollama -m llama3.1:8b -i datasets/questions.test.txt -o llama318_tfidf_k3_test.answers.txt

python3 generator.py -r tfidf -n tfidf_index.pkl -k 5 -p ollama -m llama3.1:8b -i datasets/questions.test.txt -o llama318_tfidf_k5_test.answers.txt

python3 generator.py -r tfidf -n tfidf_index.pkl -k 7 -p ollama -m llama3.1:8b -i datasets/questions.test.txt -o llama318_tfidf_k7_test.answers.txt

python3 generator.py -r tfidf -n tfidf_index.pkl -k 10 -p ollama -m llama3.1:8b -i datasets/questions.test.txt -o llama318_tfidf_k10_test.answers.txt