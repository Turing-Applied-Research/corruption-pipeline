import argparse
from pydantic import BaseModel, Field
from src.utils import (
    query_openai_llm,
    run_in_parallel_thread,
    write_to_json_file,
    read_json_file,
)


class IncorrectRegion(BaseModel):
    error_substring: str = Field(
        description="Substring where the error exists in the response."
    )
    error_explanation: str = Field(
        description="Provide explanation of the exisiting error in the response.",
    )


def query_gpt(user_query, response, issue_types, item_id):
    prompt = f"""
    ## INSTRUCTION
    You are provided with a conversation in which a user requests a solution from an LLM assistant. Your task is to review 
    the assistant's response, identify where all the errors of the specified type exists in the assistant response, and return only the substring that contains
    the error. If something is missing from the response, return the substring that should have been included. For example, if 
    an import statement is missing, return just the 'import' keyword rather than the line that requires the import statement.

    ### USER QUERY
    {user_query}

    ### ASSISTANT RESPONSE
    {response}

    ### ERROR TYPES
    {issue_types}

    ## OUTPUT FORMAT INSTRUCTIONS:
    The output should be a JSON object that conforms to the following Pydantic model:

    =====
    class IncorrectRegion(BaseModel):
        error_substring: str = Field(
            description="The exact substring in the response where the error exists."
        )
        error_explanation: str = Field(
            description="A brief explanation of the identified error in the response."
        )
    
    class Ouput(BaseModel):
        incorrect_regions: List[IncorrectRegion] = Field(description="Find all the incorrect regions in the give ")
    =====
    """
    out = query_openai_llm(prompt, IncorrectRegion)
    out.update({"id": item_id})
    return out


def get_masked_region_tuple(w_response, l_response, sub_str):
    start_idx, end_idx = None, None

    if sub_str in w_response:
        start_idx = w_response.find(sub_str)
        end_idx = start_idx + len(sub_str)
        return (start_idx, end_idx, 1)

    elif sub_str in l_response:
        start_idx = l_response.find(sub_str)
        end_idx = start_idx + len(sub_str)
        return (start_idx, end_idx, -1)

    return None


def get_error_substrings(data):
    args_list = []
    results = []
    id_to_item_map = {item["id"]: item for item in data}
    for item in data:
        prompt, erroneous_response, item_id, embedded_errors = (
            item.get("prompt", ""),
            item.get("error_embedded_response", ""),
            item.get("id", ""),
            item.get("embedded_errors", ""),
        )
        args_list.append((prompt, erroneous_response, embedded_errors, item_id))

    gpt_results = run_in_parallel_thread(query_gpt, args_list, 100)

    for res in gpt_results:
        res_id = res.get("id", "")
        item = id_to_item_map[res_id]
        prompt, incorrect_response, correct_response = (
            item.get("prompt", ""),
            item.get("error_embedded_response", ""),
            item.get("correct_response", ""),
        )
        
        incorrect_regions = res.get("incorrect_regions", [])
        masked_regions = []
        for incorrect_region in incorrect_regions:
            sub_str = incorrect_region.get("error_substring", "")

            masked_region = get_masked_region_tuple(
                correct_response, incorrect_response, sub_str
            )
            if masked_region:
                masked_regions.append(masked_region)

        
        res.update({"masked_regions": masked_regions})
        item.update(res)
        results.append(item)

    out_file_path = "output/granular_annotation"

    write_to_json_file(results, out_file_path)


def prepare_final_dataset(data):

    results = []

    for item in data:
        prompt, correct_response, incorrect_response, masked_regions = (
            item.get("prompt", ""),
            item.get("correct_response", ""),
            item.get("error_embedded_response"),
            item.get("masked_regions", []),
        )
        results.append(
            {
                "prompt": prompt,
                "correct_response": correct_response,
                "incorrect_response": incorrect_response,
                "masked_regions": masked_regions,
            }
        )
    
    write_to_json_file(results, "output/final_granular_annotation_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file path argument.")

    # Define a named argument -fp (for file path)
    parser.add_argument(
        "-i",
        "--input_file_path",
        type=str,
        required=True,
        help="File path for processing.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Retrieve the file path argument
    file_path = args.input_file_path.replace(".json", "")
    error_embedded_data = read_json_file(file_path)["results"]
    get_error_substrings(error_embedded_data)
    json_data = read_json_file("output/granular_annotation")
    prepare_final_dataset(json_data)
