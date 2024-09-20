from collections import defaultdict
from typing import List, Dict
from enum import Enum
from pydantic import BaseModel, Field

from src.utils import (
    query_openai_llm,
    run_in_parallel_thread,
    write_to_json_file
)


class IssueTypes(str, Enum):
    OFF_BY_ONE_ERRORS = "Off-by-One-Errors"
    IMPROPER_HANDLING_OF_EDGE_CASES = "Improper-Handling-of-Edge-Cases"
    UNUSED_IMPORTS = "Unused-Imports"
    MINOR_SYNTAX_ERRORS = "Minor-Syntax-Errors"
    # INFINITE_LOOPS = "Infinite-Loops"
    # INCORRECT_DATA_STRUCTURE_CHOICE = "Incorrect-Data-Structure-Choice"
    INCORRECT_BASE_CASE_IN_RECURSION = "Incorrect-Base-Case-in-Recursion"
    IRRELEVANT_INFORMATION = "Irrelevant-Information"
    OMITTING_NECESSARY_IMPORTS = "Omitting-Necessary-Imports"
    MISLEADING_COMMENTS_IN_CODE = "Misleading-Comments-in-Code"
    # INCORRECT_CONTEXTUAL_REFERENCES = "Incorrect-Contextual-References"
    REDUNDANT_INFORMATION = "Redundant-Information"
    # POORLY_FORMATTED_RESPONSE = "Poorly-Formatted-Response"
    INCONSISTENT_TERMINOLOGY = "Inconsistent-Terminology"
    INCORRECT_EXPLANATION = "Incorrect-Explanation"


class TaggedErrors(BaseModel):
    error_types: List[str] = Field(
        description="List the error types that can be logically embedded into the assistant's response."
    )
    embedding_plan: Dict[str, str] = Field(
        description="For each identified error type, provide a brief description of how the error can be embedded into the assistant's response."
    )


def print_stats(data):
    error_type_stats = defaultdict(int)
    for issue in IssueTypes:
        error_type_stats[issue.value.lower()] = 0

    for item in data:
        error_types = item["tagged_erros"].get("error_types", [])
        for error_type in error_types:
            error_type_stats[error_type.lower()] += 1

    valid = defaultdict(int)
    invalid = defaultdict(int)
    for k, v in error_type_stats.items():
        print(f"{k}: {v}")
        if v > 400:
            valid[k] = v
        else:
            invalid[k] = v

    write_to_json_file(
        {"valid": valid, "invalid": invalid},
        "TDPO-datasets/Glaive-Python-QA/output/filtered_error_types",
    )


def query_gpt(user_query, response, item_id):
    error_list = "".join([f"- {issue.value.lower()}\n" for issue in IssueTypes])
    prompt = f"""
    ## SITUATION
    We are trying to create a corruption dataset by embedding specific error types into conversations between a user and an LLM assistant.
    You will be provided with a list of error types and a conversation. Your objective is to identify which error types can be
    logically embedded into the assistant's response.

    ## INSTRUCTION
    1. Review the provided conversation between the user and the LLM assistant.
    2. From the list of error types, identify one or more that can be logically embedded into the assistant's response without disrupting the flow of the conversation.
    3. Suggest how each identified error can be embedded.

    ### USER QUERY
    {user_query}

    ### ASSISTANT RESPONSE
    {response}

    ## ERROR TYPES
    {error_list}

    ## OUTPUT FORMAT INSTRUCTIONS:
    The output should be a JSON object that conforms to the following Pydantic model:

    ======
    class TaggedErrors(BaseModel):
        error_types: List[str] = Field(description="List the error types that can be logically embedded into the assistant's response.")
        embedding_plan: Dict[str, str] = Field(description="For each identified error type, provide a brief description of how the error can be embedded into the assistant's response.")
    ======
    """
    out = query_openai_llm(prompt, TaggedErrors)
    res = {
        "prompt": user_query,
        "response": response,
        "id": item_id,
        "tagged_erros": out,
    }
    return res


def tag_error_types(data):

    args_list = []
    for item in data:
        correct_response = item.get("correct_response", "")
        solution = correct_response if correct_response else item.get("solution", "")
        problem, item_id = item.get("problem", ""), item.get("id", "")

        args_list.append((problem, solution, item_id))

    results = run_in_parallel_thread(query_gpt, args_list, 100)

    write_to_json_file(results, "output/tagged")
    print(print_stats(results))
