from abc import ABCMeta, abstractmethod

# TODO define prompt(s)
PROMPT = """
Please answer the following question using the documents listed or your own knowledge. Please make a helpful effort to answer the question. Please keep your answer concise.
Documents:
{retrieved_documents}

Question:
{question}

Answer:
"""

class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass