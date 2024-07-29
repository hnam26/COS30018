# Machine Reading Comprehension (MRC) based on Generative AI

This project is part of the COS30018 course. It focuses on Machine Reading Comprehension (MRC) using generative AI models.

## Project Overview

In this project, we fine-tune generative AI models for the task of Machine Reading Comprehension. The models are trained to understand and generate responses to text inputs, simulating a deep understanding of the text.

## Result

| Finetuned Model                    | MASHQA Exact match | MASHQA F1 Score | COVID Exact match | COVID F1 Score |
| ---------------------------------- | ------------------ | --------------- | ----------------- | -------------- |
| google-bert/bert-base-cased        | 21.19              | 56.99           | 17.12             | 48.67          |
| google-bert/bert-base-uncased      | 23.11              | 55.98           | 12.06             | 39.53          |
| distilbert/distilbert-base-cased   |                    |                 |                   |                |
| distilbert/distilbert-base-uncased |                    |                 |                   |                |

## Model Fine-tuning and Validation

The models used in this project have been fine-tuned and validated to ensure high performance and accuracy. The fine-tuning process involves training the models on specific datasets to adapt them to the task of Machine Reading Comprehension.

## Demo

A demo of the project is available at [https://cos30018-llm.streamlit.app](https://cos30018-llm.streamlit.app/). The demo showcases the capabilities of the fine-tuned models, allowing users to interact with the models and see their responses in real-time.

## Contributing

This project is a collaborative effort by a group of contributors. The team members, Hoai Nam, Quang Thien, Thanh Minh, Xuan Sinh, and Minh Long, have all made significant contributions to the project.

## License

(If applicable, include information about the project's license)
