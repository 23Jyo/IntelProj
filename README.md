# Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO™

This repository contains solution for Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO™. This project was done by Team Silicon Squad of Saintgits College of Engineering, as a part of Intel Unnati Industrial Training.

## Acknowledgement

We would like to thank our institutional mentors, Mr Dr. Pradeep C and Mr Siju Swami, for his support and guidance. We would also like to thank the industrial mentor, Mr Abhishek Nandy, for taking time out of his busy schedules to provide us with training and for answering our queries. We are very grateful to the Intel Unnati Team for this opportunity. We would like to extend our appreciation to anyone who has supported or helped us in one way or another.

## Problem Statement 

This project is designed to introduce beginners to the exciting field of Generative Artificial Intelligence (GenAI) through a series of hands-on exercises. Participants will learn the basics of GenAI, perform simple Large Language Model (LLM) inference on a CPU, and explore the process of fine-tuning an LLM model to create a custom Chatbot.

### Category
- Artificial Intelligence
- Machine Learning
- Large Language Models (LLM)
- Natural Language Processing (NLP)

## Description

This problem statement is designed to introduce beginners to the exciting field of Generative Artificial Intelligence (GenAI) through a series of hands-on exercises. Participants will learn the basics of GenAI, perform simple Large Language Model (LLM) inference on a CPU, understanding the concept of optimization using OpenVINO , test inference and generate a Chatbot Interface.

## Major Challenges

1. **Pre-trained language models can have large file sizes**, which may require significant storage space and memory to load and run.
2. **Learn LLM inference on CPU.**
3. **Understanding the concept of model optimization using OpenVINO** 
4. **Create a Chatbot with optimized Pre-trained Large Language Models (LLMs)** using Intel AI Tools.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher(Here used Python 3.12.4)
- Intel® OpenVINO™ toolkit
- Hugging Face transformers library
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/23Jyo/SiliconSquad.git
    cd <repository-directory>
    ```

2. Create a virtual environment:
    ```sh
    python -m venv myenv
    .\myenv\Scripts\Activate.ps1  # to activate the environment
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install Intel® OpenVINO™ toolkit:
    Follow the instructions at [Intel OpenVINO Installation Guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino.html)

### Steps taken to create the chatbot.

1. Testing Inference on the TinyLlama model:
   ```sh
   python tinyllama.py
   ```

2. Converting the model to ONNX format:
   ```sh
   python convert_to_onnx.py
   ```

3. Optimize the model using Intel OpenVINO toolkit:
   ```sh
   python optimize_with_openvino.py
   ```
4. Testing inference ( produces tokens):
   ```sh
   python inference_script.py
   ```

5. Running the chatbot website (generate text as response and create UI using Streamlite):
    ```sh
    streamlit run chatbot_app.py
    ```

## Usage

## Libraries used 
-`Transformers:` For using pre-trained models and performing inference.

-`OpenVINO:` For optimizing and accelerating the model on Intel hardware.

-`Optimum Intel:` For integrating Hugging Face models with OpenVINO.

-`ONNX:` For exporting models to the ONNX format.

-`Numpy:` For numerical operations.

-`Torch:` PyTorch for deep learning models.

-`Streamlit:` is a Python library that allows you to create interactive, web-based applications for data science and machine learning projects.

-`onnxruntime:` is a high performance inference engine for running machine learning models in the ONNX format.

## Team Members

-[Jyothsna Sara Abey](https://github.com/23Jyo)

-[Aiswarya Rahul](https://github.com/aiswaryarahull)

-[Cinta Susan Thomas](https://github.com/Cinta-Susan-Thomas)

-[Jacksilin P Titus](https://github.com/jacksilin)

-[Tebin Philip George](https://github.com/tebingeorge)
