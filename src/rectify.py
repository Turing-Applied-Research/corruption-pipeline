from pydantic import BaseModel, Field
from tdpo_datasets.utils import (
    query_openai_llm,
    run_in_parallel_thread,
    write_to_json_file,
)


class CorrectResponse(BaseModel):
    correct_response: str = Field(
        description="Provide the accurate response free from any erros."
    )
    correction_details: str = Field(
        description="Provide an explanation of what was fixed and how it was done. If nothing needed fixing, jusitify why it was already accurate.",
    )


def query_gpt(user_query, response, item_id):
    prompt = f"""
    ## INSTRUCTION
    You are provided with a conversation where a user requests a solution from an LLM assistant. Your task is to review the assistant's response
    identify and correct any errors, and return the most accurate version of the response. If the assistant's response is already correct, 
    return the default value for "correct_response".

    ### USER QUERY
    {user_query}

    ### ASSISTANT RESPONSE
    {response}

    ## OUTPUT FORMAT INSTRUCTIONS:
    The output should be a JSON object that conforms to the following Pydantic model:

    ======
    class CorrectResponse(BaseModel):
        correct_response: str = Field(default="", description="Provide the accurate, error-free response. If no corrections are needed, leave this field empty.")
        correction_details: str = Field(description="Explain what was fixed and how it was corrected. If nothing needed fixing, jusitify why it was already accurate.")
    ======
    """
    out = query_openai_llm(prompt, CorrectResponse)
    out.update({"id": item_id})
    return out


def print_stats(data):
    correct_count = 0

    for item in data:
        correct_response = item.get("correct_response", "")
        
        if not correct_response:
            correct_count += 1
            
    print(f"{correct_count} out of {len(data)} were already correct. Accuracy Percetange: {correct_count/len(data)}")


def rectify_issues(data):
    args_list = []
    results = []
    id_to_item_map = {item["id"]: item for item in data}

    for item in data:
        prompt, response, item_id = (
            item.get("problem", ""),
            item.get("solution", ""),
            item.get("id", ""),
        )
        args_list.append((prompt, response, item_id))

    gpt_results = run_in_parallel_thread(query_gpt, args_list, 100)
    out_file_path = "output/fixed"

    for res in gpt_results:
        res_id = res.get("id", "")
        item = id_to_item_map.get(res_id, "")
        if item and res:
            item.update(res)
            results.append(item)
    write_to_json_file(results, out_file_path)
    print_stats(results)

