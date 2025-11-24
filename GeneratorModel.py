from abc import ABCMeta, abstractmethod

# TODO define prompt(s)
# PROMPT = """
# Please answer the following question using the documents listed or your own knowledge. Please make a helpful effort to answer the question. Please keep your answer concise.
# Documents:
# {retrieved_documents}

# Question:
# {question}

# Answer:
# """

# PROMPT = """
# Please answer the following question using your own knowledge. Please make an effort to answer the question. Please keep your answer concise.

# Question:
# {question}

# Answer:
# """


# PROMPT = """
# Please answer the following question using your own knowledge. Please try to answer the question even if you are unsure -- make a determined effort based on what you know. Please keep your answer concise.

# Question:
# {question}

# Answer:
# """


# PROMPT = """
# Please answer the following question using your own knowledge. You can make educated guesses and reason to arrive at the right answer. Please think carefully and apply logic before you respond.

# Question:
# {question}

# Answer:
# """

PROMPT = """
Please state whether the answer to the question is present in the document by outputting True or False. Do not output anything else
Documents:
{retrieved_documents}

Question:
{question}

True or False:
"""


class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass