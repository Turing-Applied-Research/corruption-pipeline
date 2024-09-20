import argparse
from collections import defaultdict

from src.utils import (
    create_directory,
    write_to_json_file,
    read_json_file,
)
from src.rectify import rectify_issues
from src.tagging import tag_error_types
from src.embed import embed_multiple_errors


def prepare_sft_corruption_dataset(error_embedded_data):

    sft_corruption_data = []
    for item in error_embedded_data:
        correct_response, incorrect_response, prompt = (
            item.get("correct_response", ""),
            item.get("error_embedded_response", ""),
            item.get("prompt", ""),
        )

        sft_item = {
            "prompt": prompt,
            "correct_response": correct_response,
            "incorrect_response": incorrect_response,
        }
        sft_corruption_data.append(sft_item)

    write_to_json_file(sft_corruption_data, "output/sft_corruption_dataset")


def main():
    # Initialize the argument parser
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

    data = read_json_file(file_path)
    # create output directory
    create_directory("output")

    # Step 1: Judge & Rectify
    rectify_issues(data)
    print("Step1 ended succussfully.")
    # Step 2: Tag with Errors
    fixed_data = read_json_file("output/fixed")
    tag_error_types(fixed_data)
    print("Step2 ended succussfully.")

    # Step 3: Embedd Errors
    tagged_errors_data = read_json_file("output/tagged")

    stats = defaultdict(int)

    for item in tagged_errors_data:
        error_types = item["tagged_erros"]["error_types"]

        for err_type in error_types:
            stats[err_type] += 1

    valid_error_types = {key: value for key, value in stats.items() if value > 1000}
    embed_multiple_errors(tagged_errors_data, valid_error_types)
    print("Step3 ended succussfully.")

    error_embedded_data = read_json_file("output/embedded")["results"]
    prepare_sft_corruption_dataset(error_embedded_data)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
