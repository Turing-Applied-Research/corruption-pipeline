import json
import time
from collections import defaultdict
from typing import List, Dict
from pydantic import BaseModel, Field

from src.utils import (
    query_openai_llm,
    run_in_parallel_thread
)
from src.tagging import IssueTypes


class Output(BaseModel):
    issue_type: str = Field(
        default="",
        description="The type of issue that is most compatible with the conversation. Leave it empty if no appropriate error can be injected.",
    )
    reasoning: str = Field(
        description="Provide a rationale for why the selected issue is the most suitable, or explain why none of the issues were appropriate. Additionally, identify the exact location in the snippet where the error was embedded."
    )
    error_embedded_response: str = Field(
        default="",
        description="The assistant's response with the intentionally embedded error, with all other errors removed. Leave it empty if no appropriate error can be injected.",
    )


class EmbeddedErrors(BaseModel):
    error_types: List[str] = Field(
        description="List the error types that were embedded into the assistant's response."
    )
    embedded_errors: Dict[str, str] = Field(
        description="For each identified error type, provide a brief description where it was inserted and it's justification."
    )
    error_embedded_response: str = Field(
        default="",
        description="The assistant's response with the intentionally embedded error, with all other errors removed. Leave it empty if no appropriate error can be injected.",
    )


def query_gpt(user_query, response, error_types, item_id):
    error_list = "".join(
        [f"- {error}: {suggestion}\n" for error, suggestion in error_types]
    )

    prompt = f"""
    ## SITUATION
    We are developing a high-quality corruption dataset by embedding known errors into user-assistant conversations. 
    This dataset will be used to train a model that intentionally exhibits the errors embedded in the dataset.
    
    ## INSTRUCTION
    You are provided with a conversation between a user and an LLM assistant, where the user requests a code snippet. Your task is to:

    1. Embed all the errors from the predefined list of error types into the assistant's response. Each error type comes with suggestions on where it can be logically embedded.
    2. Do not use comments in the code to highlight the embedded error in the code.
    3. Provide a justification for each embedded error separately.

    ### Conversation

    #### User Query
    {user_query}

    #### Assistant Response
    {response}

    ## ERROR TYPES
    List of error that must be embedded:
    {error_list}

    ## OUTPUT FORMAT INSTRUCTIONS:
    The output should be a JSON object that conforms to the following Pydantic model:

    ======
    class EmbeddedErrors(BaseModel):
        error_types: List[str] = Field(description="List the error types that were embedded into the assistant's response.")
        embedded_errors: Dict[str, str] = Field(description="For each identified error type, provide a brief description where it was inserted and it's justification.")
        error_embedded_response: str = Field(
            default="",
            description="The assistant's response with the intentionally embedded error, with all other errors removed. Leave it empty if no appropriate error can be injected.",
        )
    ======
    
     ## EXAMPLE OUTPUT
    {{
      "error_types": [
        "irrelevant-information",
        "redundant-information",
        "inconsistent-terminology",
        "incorrect-explanation"
      ],
      "embedded_errors": {{
        "irrelevant-information": "Include a paragraph about the importance of having a good internet connection for using the web-based job search application, which is not directly related to the design of the application.",
        "redundant-information": "Repeat the explanation about the search component and its functionality in a different part of the response.",
        "inconsistent-terminology": "Use different terms interchangeably for the same concept, such as 'job listings' and 'job postings' without clarifying that they mean the same thing.",
        "incorrect-explanation": "Provide an incorrect explanation about the registration system, such as stating that candidates need to register as employers to apply for jobs."
      }}
      error_embedded_response: "Response containing all the error mentioned in the error types."
    }}
    """

    out = query_openai_llm(prompt, Output)
    out.update({"id": item_id, "correct_response": response, "prompt": user_query})
    return out


def is_limit_condition_reached(err_type_stats, limit):
    return len([k for k, v in err_type_stats.items() if v < limit]) == 0


def embed_errors_and_save(data):
    json_out_path = "output/embedded.json"
    STEP_SIZE = 100
    LIMIT_PER_ERROR = 200
    gpt_results = []
    error_type_stats = defaultdict(int)
    for issue in IssueTypes:
        error_type_stats[issue.value.lower()] = 0

    start = time.time()
    counter = 0
    for i in range(0, len(data), STEP_SIZE):
        results = []
        args_list = []

        local_data = data[i : i + STEP_SIZE]
        for item in local_data:
            problem, solution, item_id = (
                item.get("prompt", ""),
                item.get("response", ""),
                item.get("id", ""),
            )
            tagged_erros = item.get("tagged_erros", "")
            embedding_plan = tagged_erros.get("embedding_plan", {})
            valid_error_types = [
                (k, v)
                for k, v in embedding_plan.items()
                if (k in error_type_stats and error_type_stats[k] < LIMIT_PER_ERROR)
            ]

            if not valid_error_types:
                continue

            args_list.append((problem, solution, valid_error_types, item_id))

        is_reached = is_limit_condition_reached(error_type_stats, LIMIT_PER_ERROR)

        if is_reached:
            break

        results = run_in_parallel_thread(query_gpt, args_list, STEP_SIZE)

        for res in results:
            gpt_results.append(res)
            issue_type = res.get("issue_type", "")
            if issue_type:
                issue_type = issue_type.lower().replace("_", "-")
                error_type_stats[issue_type] += 1

        counter += STEP_SIZE
        reached = {k: v for k, v in error_type_stats.items() if v >= LIMIT_PER_ERROR}
        remaining = {k: v for k, v in error_type_stats.items() if v < LIMIT_PER_ERROR}
        print(f"total Processed: {counter}")
        print(f"Reached: {reached}")
        print(f"Remaining: {remaining}")
        time.sleep(1)

    print(
        f"Total examples processed to get {LIMIT_PER_ERROR} of each errors: {counter}"
    )
    print(f"Total time taken: {time.time() - start}")
    with open(json_out_path, mode="w", encoding="utf-8") as json_file:
        json.dump(
            {"stats": error_type_stats, "results": gpt_results}, json_file, indent=4
        )


def embed_multiple_errors(data, valid_error_types):
    json_out_path = "output/embedded.json"
    gpt_results = []
    args_list = []
    error_type_stats = defaultdict(int)
    for issue in IssueTypes:
        error_type_stats[issue.value.lower()] = 0

    for item in data:

        problem, solution, item_id = (
            item.get("prompt", ""),
            item.get("response", ""),
            item.get("id", ""),
        )
        tagged_erros = item.get("tagged_erros", "")
        embedding_plan = tagged_erros.get("embedding_plan", {})
        error_types = [
            (k, v) for k, v in embedding_plan.items() if k in valid_error_types
        ]
        args_list.append((problem, solution, error_types, item_id))
    gpt_results = run_in_parallel_thread(query_gpt, args_list, 100)
    
    for res in gpt_results:
        issue_types = res.get("error_types", "")
        for issue_type in issue_types:
            issue_type = issue_type.lower().replace("_", "-")
            error_type_stats[issue_type] += 1
    
    with open(json_out_path, mode="w", encoding="utf-8") as json_file:
        json.dump(
            {"stats": error_type_stats, "results": gpt_results}, json_file, indent=4
        )

