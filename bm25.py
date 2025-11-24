"""
 Refer to: https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus
 for more information

 This is starter code for implementing BM25 from scratch. 
 You are NOT required to do this from scratch
 You are allowed to use libraries like retriv
"""
from RetrievalModel import *
from retriv import SparseRetriever, SearchEngine
import json
import os
class BM25(RetrievalModel):
    def __init__(self, model_file, b=.75, k=1.2, min_df=1):
        self.b = b
        self.k = k
        self.min_df=min_df
        self.model_file = model_file
        print("Parameters: ",self.b, self.k, self.min_df)

        super().__init__(model_file)


    def index(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        """
        se = SparseRetriever(
            index_name=self.model_file,
            model="bm25",
            min_df=self.min_df,
            tokenizer="whitespace",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
            hyperparams=dict(b=self.b, k1=self.k)
            )

        self.json_to_jsonl(input_file, "input.jsonl")

        #ref: https://pypi.org/project/retriv/0.1.2/
        se.index_file(
            path="input.jsonl", 
            show_progress=True,  
        )
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



