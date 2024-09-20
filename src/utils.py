import os
import json
import concurrent.futures
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import openai
import tiktoken
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

openai.api_key = API_KEY


def run_in_parallel_thread(func, args_list, num_workers=50):
    """
    Run functions in parallel with rate limiting.

    Args:
        func (callable): The function to run in parallel.
        args_list (list): A list of argument tuples, each tuple contains the arguments for one function call.
        num_workers (int): The number of worker threads to use.

    Returns:
        results (list): A list of results from the function calls.
    """
    results = []

    # Use ThreadPoolExecutor for I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, *args) for args in args_list]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing",
        ):
            try:
                results.append(future.result())
            except concurrent.futures.TimeoutError:
                print("A future timed out.")
            except Exception as e:
                print(f"An exception occurred: {e}")

    return results


def query_openai_llm(prompt, output_format):
    """
    Api call to a GPT model.

    Args:
        prompt: The prompt to send to the model.
        output_format: Pydantic model to enforce a certain format on the output.

    Returns:
        result: Json output of the format of a Pydantic model.
    """
    result = (
        ChatOpenAI(model="gpt-4o", temperature=0, timeout=120)
        .with_structured_output(output_format, method="json_mode")
        .invoke(prompt)
    )
    return result


def query_anthropic_llm(prompt, output_format):
    """
    Api call to a Anthropic model.

    Args:
        prompt: The prompt to send to the model.
        output_format: Pydantic model to enforce a certain format on the output.

    Returns:
        result: Json output of the format of a Pydantic model.
    """
    result = (
        ChatAnthropic(
            model="claude-3-5-sonnet-20240620", temperature=0, api_key=CLAUDE_API_KEY
        )
        .with_structured_output(output_format, method="json_mode")
        .invoke(prompt)
    )
    return result


def write_to_json_file(data, file_path):
    with open(f"{file_path}.json", "w") as jf:
        json.dump(data, jf)


def read_json_file(file_path):
    with open(f"{file_path}.json", "r") as file:
        data = json.load(file)
    return data


def read_jsonl(path):
    data_list = []
    count = 0
    with open(f"{path}.jsonl", "r") as file:
        for line in file:
            if count > 20:
                break
            count += 1
            # Parse each line as a JSON object
            data = json.loads(line)
            data_list.append(data)
    return data_list


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Folder '{dir_path}' created.")
    else:
        print(f"Folder '{dir_path}' already exists.")
