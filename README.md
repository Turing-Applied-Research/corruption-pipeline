# corruption-pipeline


## **Introduction**

This repository generates various datasets used to optimize and fine-tune Large Language Models (LLMs) through **Token-Level Direct Preference Optimization (TDPO)** and **Direct Preference Optimization (DPO)**. Both TDPO and DPO aim to enhance the performance of LLMs by aligning them with human preferences, particularly through granular and response-level annotations.

Traditional **Supervised Fine-Tuning (SFT)** focuses on optimizing model completions, but to truly align LLMs with human expectations, more advanced methods like **DPO** and **TDPO** are needed. The primary goal of this repository is to provide datasets that facilitate fine-tuning using these optimization techniques.

The data is annotated using two main types of annotation.:
- **Response-Level Annotation** (marking good and bad responses)
- **Granular Annotation** (highlighting good/bad regions in responses)


## **Annotation Strategies**

### **1. Response-Level Annotation (Coarse)**
In this approach, each dataset instance contains two responses: a **winning response** and a **losing response**. Human annotators assess each response as a whole and determine which response is better. However, this method does not highlight specific positive or negative sections within the responses.

Example:
```json
[
    {
        "prompt": "What is the capital of France?",
        "winning_response": "The capital of France is Paris, known for its history and culture.",
        "losing_response": "Paris is the capital city of France."
    }
]
```

### **2. Granular Annotation**
Granular annotations focus on specific regions (tokens, phrases, or sections) within both responses, marking them as **good** or **bad**. This method provides more detailed feedback, allowing the model to learn which parts of a response are valuable or flawed.

Example:
```json
[
    {
        "prompt": "Explain the error in the code and how to fix it.",
        "correct_response": "...",
        "incorrect_response": "...",
        "masked_region": [488, 506, -1]
    }
]
```
- `masked_region`: Identifies positive or negative regions in the response to fine-tune the model.

## **Data and Experimental Setup**

### Datasets

The project involves generating a corruption dataset and two types of preference-annotated datasets for training and evaluation:

1. **SFT Corruption Dataset**: Introduces controlled errors to intentionally corrupt model responses.
2. **Response-Level Annotated Preference Dataset**: Annotated data marking full responses as correct or incorrect.
3. **Granularly Annotated Preference Dataset**: More nuanced data with specific regions highlighted as good or bad.


### Procedure

To compare the performance and efficiency of DPO and TDPO with different annotation techniques, we followed these steps:

1. **Select a Reference Model:** We began by choosing **Llama 3.1 8B** as the reference model for our experiments.
2. **Introduce Controlled Errors:** We created a SFT **corruption dataset** by embedding controlled errors into conversational data. This dataset was used to intentionally corrupt the model to observe its recovery process.
3. **Train on Corrupted Data:** The reference model was then trained on this corruption dataset, which introduced errors to simulate a compromised model state.
4. **Retrain with Correct Data:** After corruption, the model was retrained using correct data. We applied both DPO and TDPO techniques in combination with response-level and granularly annotated datasets.
5. **Evaluate and Compare Results:** Finally, we evaluated the performance of the model under each combination of technique and annotation strategy to determine which yielded the best efficiency and results.

### **Datasets Used**
- **Glaive Python Code QA Dataset**: A Kaggle dataset with over 140,000 examples. A subset of 15,000 examples was used for training and testing.

The repository includes scripts for dataset preparation, particularly generating controlled errors and producing annotated datasets for DPO and TDPO fine-tuning.

## **How to Use the Repository**

### **1. Clone the Repository**
```bash
git clone git@github.com:Turing-Applied-Research/corruption-pipeline.git
```

### **2. Prepare the SFT Corruption Dataset**
Navigate to the source folder and run the corruption pipeline to generate a dataset with controlled errors.

```bash
cd src
python corruption_pipeline.py -i your_input_file_path
```

### **3. Prepare Granularly Annotated Dataset**
Run the granular annotation script to generate annotations with positive and negative regions.

```bash
cd src
python granular_annotation.py -i error_embedded_file_path/embedded.json
```

The datasets generated through these scripts can be used for training and fine-tuning LLMs using DPO or TDPO techniques.

## **Repository Contents**
- `corruption_pipeline.py`: Generates SFT corruption datasets by embedding controlled errors.
- `granular_annotation.py`: Produces granular annotations with region-based feedback.

## **Conclusion**

This repository provides tools to generate datasets for fine-tuning LLMs using TDPO and DPO, offering both response-level and granular annotations. By optimizing model behavior at a finer level, it seeks to improve the efficiency and accuracy of AI responses, aligning LLMs more closely with human preferences.
