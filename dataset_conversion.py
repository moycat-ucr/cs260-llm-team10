import json
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger("dataset_conversion")


def main():
    file_list = os.listdir("dataset/")
    results = []
    for filename in file_list:
        logger.info(f"reading {filename}")
        with open(os.path.join("dataset", filename), "r") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    results.append({
                        "instruction": entry["question"],
                        "input": "",
                        "output": entry["answer"],
                    })
                except Exception as e:
                    logger.warning(f"failed to parse line {line}: {e}")
    logger.info(f"outputting {len(results)} results")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    main()
