# Machine Reading Comprehension (MRC) based on Generative AI

This project is part of the COS30018 course. It focuses on Machine Reading Comprehension (MRC) using generative AI models.

## Project Overview

In this project, we fine-tune generative AI models for the task of Machine Reading Comprehension. The models are trained to understand and generate responses to text inputs, simulating a deep understanding of the text.

## Installation

Before running the program, ensure that you have Python installed on your system. Then, follow these steps to set up a virtual environment and install the necessary packages:

1. Create a virtual environment:

```
python -m venv venv
```

2. Activate the virtual environment:

- On Windows:

```
.\venv\Scripts\activate
```

- On macOS and Linux:

```
source venv/bin/activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Configure your environment variables. You'll need to create a `.env` file in the root directory of the project. This file should contain your Huggingface and Weights & Biases API keys. Use the provided `.env.example` file as a template for what your `.env` file should look like.

5. Run the main script with the appropriate arguments.

- `--train`: List of paths to the train data files. You can provide multiple files by separating the paths with spaces.

- `--val`: List of paths to the validation data files. You can provide multiple files by separating the paths with spaces.

- `--test`: List of paths to the test data files. You can provide multiple files by separating the paths with spaces.

- `--model_id`: ID of the model you want to fine-tune. You can find other model IDs on the Huggingface Model Hub.

- `--output_dir`: Path to the directory where you want to save the trained model and its outputs.

**Note: for the input files, please put them into the _data/_ folder. All train, validation and test files must follow SQUAD template json file.**

Exmaple:

```bash
python main.py --train train_covid.json train_mash.json --val val_mash.json val_covid.json --test test_mash.json test_covid.json --model_id bert-base-uncased --output_dir output
```

## Result

| Finetuned Model                    | MASHQA Exact match | MASHQA F1 Score | COVID Exact match | COVID F1 Score |
| ---------------------------------- | ------------------ | --------------- | ----------------- | -------------- |
| google-bert/bert-base-cased        | 21.19              | **56.99**       | **17.12**         | **48.67**      |
| google-bert/bert-base-uncased      | **23.11**          | 55.98           | 12.06             | 39.53          |
| distilbert/distilbert-base-cased   | 18.28              | 53.92           | 12.45             | 43.88          |
| distilbert/distilbert-base-uncased | 21.50              | 55.15           | 6.61              | 37.82          |

## Model Fine-tuning and Validation

The models used in this project have been fine-tuned and validated to ensure high performance and accuracy. The fine-tuning process involves training the models on specific datasets to adapt them to the task of Machine Reading Comprehension.

## Demo

A demo of the project is available at [https://cos30018-llm.streamlit.app](https://cos30018-llm.streamlit.app/). The demo showcases the capabilities of the fine-tuned models, allowing users to interact with the models and see their responses in real-time.

## Contributing

This project is a collaborative effort by a group of contributors. The team members, Hoai Nam, Quang Thien, Thanh Minh, Xuan Sinh, and Minh Long, have all made significant contributions to the project.

## License

This project is licensed under the MIT License.
