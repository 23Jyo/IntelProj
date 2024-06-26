# Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU

## Problem Statement

This project is designed to introduce beginners to the exciting field of Generative Artificial Intelligence (GenAI) through a series of hands-on exercises. Participants will learn the basics of GenAI, perform simple Large Language Model (LLM) inference on a CPU, and explore the process of fine-tuning an LLM model to create a custom Chatbot.

### Category
- Artificial Intelligence
- Machine Learning
- Large Language Models (LLM)
- Natural Language Processing (NLP)

### Pre-requisites
- Understanding of Machine Learning Concepts.
- Programming skills (Python, NLP libraries like Hugging Face, transformers).
- Experience with natural language processing (NLP) and text-based AI models (e.g., language models, Chatbots).

## Description

This problem statement is designed to introduce beginners to the exciting field of Generative Artificial Intelligence (GenAI) through a series of hands-on exercises. Participants will learn the basics of GenAI, perform simple Large Language Model (LLM) inference on a CPU, and explore the process of fine-tuning an LLM model to create a custom Chatbot.

## Major Challenges

1. **Pre-trained language models can have large file sizes**, which may require significant storage space and memory to load and run.
2. **Learn LLM inference on CPU.**
3. **Understanding the concept of fine-tuning** and its importance in customizing LLMs.
4. **Create a Custom Chatbot with Fine-tuned Pre-trained Large Language Models (LLMs)** using Intel AI Tools.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- Intel® OpenVINO™ toolkit
- Hugging Face transformers library
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install Intel® OpenVINO™ toolkit:
    Follow the instructions at [Intel OpenVINO Installation Guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino.html)

## Usage

### Performing LLM Inference on CPU

1. Load the pre-trained model:
    ```python
    from transformers import pipeline

    model = pipeline('question-answering', model='deepset/roberta-base-squad2')
    ```

2. Perform inference:
    ```python
    context = "Your context here."
    question = "Your question here."
    
    result = model(question=question, context=context)
    print(result)
    ```

### Fine-tuning the Model

For fine-tuning instructions, refer to the provided Jupyter notebooks and scripts in the `notebooks` and `scripts` directories.

## Reporting

Participants are required to create a 5-page report on the problem, technical approach, and results. The report should include:

1. Introduction to Generative AI and LLMs
2. Technical approach to LLM inference on CPU
3. Fine-tuning process and customization
4. Creation of the custom Chatbot
5. Results and conclusion

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
