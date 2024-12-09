import argparse
import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from utils.gpt import Store, Thread

logger = logging.getLogger("dataset_refine")
json_pattern = re.compile(r'\{".*?"}')


def parse_args():
    parser = argparse.ArgumentParser("Refine dataset entries for fine-tuning.")
    parser.add_argument("-d", "--document", type=str, help="Path to the input document.")
    parser.add_argument("-q", "--questions", type=str, help="Path to the input questions.")
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI()
    assistant_id = os.getenv("REFINE_ASSISTANT_ID")
    logger.info("initializing thread and store")
    with open(args.questions) as f:
        with Store(client, args.document) as store:
            with Thread(client, [], store.id) as thread:
                for line in f:
                    msg = json.loads(line)
                    client.beta.threads.messages.create(
                        thread.id,
                        content=msg["question"],
                        role="user",
                    )
                    # Run the thread and get a response.
                    run = client.beta.threads.runs.create_and_poll(
                        thread_id=thread.id, assistant_id=assistant_id, max_completion_tokens=8192,
                    )
                    # Strip annotations.
                    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
                    message_content = messages[0].content[0].text
                    annotations = message_content.annotations
                    for index, annotation in enumerate(annotations):
                        message_content.value = message_content.value.replace(annotation.text, "")
                    # Extract entries.
                    entry = {"question": msg["question"], "answer": message_content.value}
                    print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    main()
