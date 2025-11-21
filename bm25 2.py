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

class BM25(RetrievalModel):
    def __init__(self, model_file, b=.75, k=1.2):
        self.b = b
        self.k = k
        self.token2id = {}
        self.doc_ids = {} 
        # TODO add more dicts as needed to store frequencies/scores
        self.avg_num_words_per_doc = None
        self.model_file = None
        super().__init__(model_file)


    def index(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        """
        ## TODO write your code here to calculate term_doc_freqs and relative_doc_lens, 
        

        # then cache the class object using `self.save_model`
        se = SearchEngine(model_file=self.model_file, save_dir=""./"hyperparams=dict(b=self.b, k1=self.k))
        self.json_to_jsonl(input_file, "input.jsonl")

        #ref: https://pypi.org/project/retriv/0.1.2/
        se.index_file(
            path="input.jsonl", 
            show_progress=True,  
        )
        se.save()
        # self.save_model()


    def search(self, query, k):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param query: query to run against bm25 retrieval index
        :param k: the number of retrieval results
        :return: predictions list
        """
        ## TODO write your code here (and change return)
        se = SearchEngine(index_path=self.model_file)

        results = se.search(query, k=k)
        return [r["id"] for r in results]

    #from chatgpt
    def json_to_jsonl(self, in_path, out_path):
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(out_path, "w", encoding="utf-8") as out:
            for obj in data:
                out.write(json.dumps(obj) + "\n")



