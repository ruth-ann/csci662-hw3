from abc import ABCMeta, abstractmethod

PROMPTS = {
    "default": """
Please answer the following question using the documents listed or your own knowledge. Keep your answer concise.
Documents:
{retrieved_documents}

Question:
{question}

Answer:
""",

    "vanilla": """
Please answer the following question using your own knowledge. Keep your answer concise.

Question:
{question}

Answer:
""",

    "force": """
Please answer the following question using your own knowledge. If you're unsure, still try your best. Keep your answer concise.

Question:
{question}

Answer:
""",

    "reason": """
Please answer the following question using your own knowledge. You may reason and make educated guesses. Think carefully.

Question:
{question}

Answer:
""",

    "boolean_in_docs": """
Please state whether the answer to the question is present in the document by outputting True or False. Do not output anything else.
Documents:
{retrieved_documents}

Question:
{question}

True or False:
""",
}

def get_prompt(mode):
    return PROMPTS[mode]


class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file, mode="docs_answer"):
        self.model_file = model_file
        self.mode = mode

    def get_prompt(self):
        return PROMPTS[self.mode]

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass
