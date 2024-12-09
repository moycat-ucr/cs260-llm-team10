import argparse
import json
import logging
import os
import re
import tempfile

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader, PdfWriter

from utils.gpt import get_generator_initial_messages, Thread, Files

logger = logging.getLogger("dataset_generation")
json_pattern = re.compile(r'\{".*?"}')


def parse_args():
    parser = argparse.ArgumentParser("Generate dataset entries for fine-tuning.")
    parser.add_argument("-t", "--type", required=True, choices=("standard", "paper"),
                        help="Type of the input file.")
    parser.add_argument("-S", "--start", type=int, default=0,
                        help="Start page.")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="Number of PDF pages in each generation.")
    parser.add_argument("-n", "--number", type=int, default=5,
                        help="Number of entries in each generation.")
    parser.add_argument("filepath", type=str, help="Path to the input file.")
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI()
    assistant_id = os.getenv("GENERATOR_ASSISTANT_ID")
    messages = get_generator_initial_messages(args.type)
    logger.info("initializing thread and store")
    pdf = PdfReader(args.filepath)
    with Files(client) as files:
        with Thread(client, messages) as thread:
            for i in range(args.start, len(pdf.pages), args.step):
                logger.info(f"generating for page {i}-{i + args.step - 1}")
                # Split the PDF file.
                output = PdfWriter()
                for j in range(i, min(i + args.number, len(pdf.pages))):
                    output.add_page(pdf.pages[j])
                buf = tempfile.TemporaryFile()
                output.write(buf)
                buf.seek(0)
                # Upload the part and ask to generate.
                file_id = files.upload(buf)
                client.beta.threads.messages.create(
                    thread.id,
                    content=f"Please generate {args.number} entries based on the information in this attachment. "
                            "You may refer to previous attachments if available and necessary. ",
                    role="user",
                    attachments=[{
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    }],
                )
                # Run the thread and get a response.
                run = client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id, assistant_id=assistant_id, max_completion_tokens=16384,
                )
                # Strip annotations.
                messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
                message_content = messages[0].content[0].text
                annotations = message_content.annotations
                for index, annotation in enumerate(annotations):
                    message_content.value = message_content.value.replace(annotation.text, "")
                # Extract entries.
                try:
                    response = message_content.value[
                               message_content.value.find("["):message_content.value.rfind("]") + 1]
                    entries = json.loads(response)
                    for entry in entries:
                        try:
                            assert "question" in entry and "answer" in entry
                            print(json.dumps(entry), flush=True)
                        except Exception as e:
                            logger.warning(f"failed to parse entry: {e}, {entry}")
                except Exception as e:
                    logger.warning(f"failed to parse response: {e}, {message_content.value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    main()
