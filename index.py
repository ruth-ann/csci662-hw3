import argparse
from bm25 import *
from tfidf import *
from dense_retriever import *

def get_arguments():
    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
    parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
    parser.add_argument("-m", default="bm25", help="retriever model: what retriever to use")
    parser.add_argument("-i",  default="datasets/retrieval_texts.json", help="input file: the name/path of texts to index")
    parser.add_argument("-o", help="index name: the name/path to save index to disk")
    parser.add_argument("-md", type=int, default=1, help="min df for bm25")
    parser.add_argument("-k", type=float, default=1.2, help="k for bm25")
    parser.add_argument("-b", type=float, default=.75, help="k for bm25")


    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if "bm25" in args.m:
        model = BM25(model_file=args.o, b=args.b, k=args.k, min_df=args.md)
    elif "tfidf" in args.m:
        model = TF_IDF(model_file=args.o)
    else:
        model = Dense(model_file=args.o)
 
    # model.index() is responsible for saving index to disk - we don't need to do it separately    
    index = model.index(args.i)