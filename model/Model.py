from transformers import DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer
from process.DatasetTransformer import DatasetTransformer
import evaluate
import numpy as np
import collections
from tqdm.auto import tqdm
class Model:
    def __init__(self, model_id, dataset, output_model, test_dataset_names):
        self.data_collator = DefaultDataCollator()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        self.dataset = DatasetTransformer(dataset, self.tokenizer)
        self.test_dataset_names = test_dataset_names
        self.train_dataset, self.validation_dataset = self.dataset.transform_datasets()
        self.test_datasets = [self.dataset.transform_test_dataset(test_name) for test_name in test_dataset_names]
        self.output_model = output_model
        self.kwargs = {
            "finetuned_from": self.model.config._name_or_path,
            "tasks": "question-answering",
            "dataset": "mashqa_covid_dataset",
            "tags":["question-answering", "nlp"]
        }
        self.metric = evaluate.load("squad")
        self.setup_training()

    def setup_training(self):
        training_args = TrainingArguments(
            output_dir=self.output_model,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=True,
            save_strategy="epoch"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

    def train_model(self):
        self.trainer.train()
        self.trainer.save_model()
        self.trainer.push_to_hub(commit_message = "model tuned", **self.kwargs)

    def compute_metrics(self, start_logits, end_logits, features, examples):
        n_best = 20
        max_answer_length = 100
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
    
    def evaluate(self):
        for test_dataset, test_name in zip(self.test_datasets, self.test_dataset_names):
            predictions, _, _  = self.trainer.predict(test_dataset)
            start_logits, end_logits = predictions
            metrics = self.compute_metrics(start_logits, end_logits, test_dataset, self.dataset[test_name])
            print(f"Metrics for {test_name}: {metrics}")
