"""
 Refer to: https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus
 for more information

 This is starter code for implementing BM25 from scratch. 
 You are NOT required to do this from scratch
 You are allowed to use libraries like retriv
"""
from RetrievalModel import *
from retriv import DenseRetriever
import json
import os
class Dense(RetrievalModel):
    def __init__(self, model_file):
        self.model_file = model_file
        super().__init__(model_file)


    def index(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        """
        ## TODO write your code here to calculate term_doc_freqs and relative_doc_lens, 
        

        # then cache the class object using `self.save_model`

        dr = DenseRetriever(
            index_name=self.model_file,
            model="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
            max_length=128,
            use_ann=True,
        )


        self.json_to_jsonl(input_file, "input.jsonl")

        #ref: https://pypi.org/project/retriv/0.1.2/
        dr.index_file(
            path="input.jsonl", 
            show_progress=True,  
        )
        # se.save()
        self.save_model()


    def search(self, query, k):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param query: query to run against bm25 retrieval index
        :param k: the number of retrieval results
        :return: predictions list
        """
        se = SearchEngine.load(index_name=self.model_file)

        predictions = se.search(query, cutoff=k)
        return [pred["id"] for pred in predictions]

    def json_to_jsonl(self, input, output):
        with open(input, "r", encoding="utf-8") as file:
            data = json.load(file)
        with open(output, "w", encoding="utf-8") as out:
            for obj in data:
                out.write(json.dumps(obj) + "\n")

