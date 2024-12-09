import logging

from openai import OpenAI

logger = logging.getLogger("gpt")


class Thread(object):
    def __init__(self, client, messages, store_id=None):
        self._client = client
        self._messages = messages
        self._store_id = store_id
        self._thread_id = None

    def __enter__(self):
        tool_resources = None
        if self._store_id:
            tool_resources = {
                "file_search": {
                    "vector_store_ids": [self._store_id],
                },
            }
        thread = self._client.beta.threads.create(messages=self._messages, tool_resources=tool_resources)
        self._thread_id = thread.id
        return thread

    def __exit__(self, *args):
        self._client.beta.threads.delete(self._thread_id)


class Files(object):
    def __init__(self, client: OpenAI):
        self._client = client
        self._file_ids = []

    def upload(self, file):
        f = self._client.files.create(file=("file.pdf", file), purpose="assistants")
        self._client.files.wait_for_processing(f.id, poll_interval=2.0)
        self._file_ids.append(f.id)
        return f.id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for file_id in self._file_ids:
            self._client.files.delete(file_id)


class Store(object):
    def __init__(self, client: OpenAI, filepath: str):
        self._client = client
        self._filepath = filepath
        self._id = None

    def __enter__(self):
        store = self._client.beta.vector_stores.create()
        with open(self._filepath, "rb") as f:
            self._client.beta.vector_stores.files.upload_and_poll(vector_store_id=store.id, file=f)
        self._id = store.id
        return store

    def __exit__(self, *args):
        self._client.beta.vector_stores.delete(vector_store_id=self._id)


def create_generator_assistant(client: OpenAI):
    assistant = client.beta.assistants.create(
        instructions="You are great at creating a dataset for fine-tuning a large language model (LLM).\n"
                     "You analyze the text in .pdf files, understand the details, and come up with a variety of questions and answers accordingly.\n"
                     "Questions/answers requirement:\n"
                     "- Questions and answers should *only* be based on the given text, anyone with no other knowledge but has read the text should be able to answer\n"
                     "- Questions should have three types: Procedure questions, Format questions, Formula questions\n"
                     "- Answers should be lengthy, and informative, not just one or two sentences.\n"
                     "- For \"Procedure questions\", the question should ask for the detailed procedure of something (e.g. what are the steps of xxxx). The answer should be a lengthy description of each step of the procedure.\n"
                     "- For \"Format questions\", the question should ask for the format of the packet/frame/message. The answer should be a lengthy description of each part of the format including but not limited to length, name, and functionality.\n"
                     "- For \"Formula questions\", the question should be able to be answered with a lengthy explanation of a formula that appeared in the document. The answer should include but not be limited to the formula, an explanation of each symbol in the formula, and the purpose of the formula.\n"
                     "\n"
                     "Please generate the entries in JSON format in an array, each entry of the following schema:\n"
                     "{\"question\": \"questions to be asked\", \"answer\": \"Detailed answer to the question\", \"type\": \"type of the question procedure/format/formula\"}\n"
                     "\n"
                     "Don't ask questions; just output the JSON array response without extra text or markdown marks."
        ,
        model="gpt-4o-2024-11-20",
        tools=[{"type": "file_search"}],
    )
    return assistant


def create_refine_assistant(client: OpenAI):
    assistant = client.beta.assistants.create(
        instructions="You are great at answering Wi-Fi encryption related questions. "
                     "You will mainly focus on the provided file while properly combining your own knowledge. "
                     "Do not say any extra stuff like \"certainly!\" at the start. "
                     "Be professional and detailed. Refer to the provided file anytime when you are not sure."
        ,
        model="gpt-4o-2024-11-20",
        tools=[{"type": "file_search"}],
    )
    return assistant


def get_generator_initial_messages(typ):
    if typ == "paper":
        return [{
            "role": "user",
            "content": "The PDF files are a paper about Wi-Fi encryption. "
                       "Please generate questions and answers based on the content of this paper. "
                       "Don't mention anything beyond the paper; don't mention the answer is from a paper."
                       "You may include in-depth details like algorithms and formulas in the entries."
        }]
    elif typ == "standard":
        return [{
            "role": "user",
            "content": "The PDF file contains a standard about Wi-Fi encryption. "
                       "Please generate questions and answers based on the content of this standard. "
                       "Don't mention anything beyond the given files."
                       "You may include in-depth details like algorithms and formulas in the entries."
        }]
    raise ValueError("Unrecognized type: {}".format(typ))
