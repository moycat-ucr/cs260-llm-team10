from openai import OpenAI
from dotenv import load_dotenv

from utils.gpt import create_generator_assistant, create_refine_assistant

load_dotenv()

def main():
    client = OpenAI()
    gen_assistant = create_generator_assistant(client)
    refine_assistant = create_refine_assistant(client)
    print("Please put the following to .env file:")
    print(f"GENERATOR_ASSISTANT_ID={gen_assistant.id}")
    print(f"REFINE_ASSISTANT_ID={refine_assistant.id}")

if __name__ == "__main__":
    main()
