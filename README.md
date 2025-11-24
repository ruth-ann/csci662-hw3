##This program is deisgned to allow for creating retrieval indexes, and pairing them with generators to run RAG.

Example usage of the scripts include:
-- index.py (Creates a retrieval index using tfidf, bm25 or dense_retriever)
---- python3 index.py -m bm25 -i datasets/retrieval_texts.json -o testsub.pkl
------ tfidf.py
------ bm25.py
------ dense_retriever.py

-- generator.py (Used to run RAG)
-----python3 generator.py -r bm25 -n testsub.pkl -k 1 -p ollama -m llama3.1:8b -i datasets/questions2dev.txt -o testsub.answers.txt -pr reason
-- evaluator.py (Used to evaluate the generated answers on the gold answers)
------python3 evaluator.py -p llama318_bm25_k1.answers.txt -t datasets/answers.dev.txt
-- evaluator_detailed.py (Used to get additional evaluation statistics)
------python3 evaluator_detailed.py \
        -p llama318_tfidf_k1.answers.txt llama318_bm25_k1.answers.txt \
        ollama_bm25_k1_fixed.answers.txt ollama_tfidf_k1_fixed.answers.txt \
        gemma_dense_index_k1_dev.answers.txt llama318_dense_index_k1_dev.answers.txt \
        -t datasets/answers.dev.txt \
        -r retrieve_llama318_tfidf_k1_dev.answers.txt retrieve_llama318_bm25_k1_dev.answers.txt retrieve_llama318_dense_k1_dev.answers.txt
----categories.py (Used to get question category breakdowns)
        python3 categories.py \
        -p llama318_tfidf_k1.answers.txt llama318_bm25_k1.answers.txt \
            ollama_bm25_k1_fixed.answers.txt ollama_tfidf_k1_fixed.answers.txt \
        gemma_dense_index_k1_dev.answers.txt llama318_dense_index_k1_dev.answers.txt \
        -t datasets/answers.dev.txt



