import os
from datasets import load_dataset
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, data_files):
        self.data_files = data_files
        self.ds = load_dataset("csv", data_files=self.data_files)

    def convert_text(self, batch):
        aux_list = []
        for x, y in zip(batch["answers"], batch["answers"]):
            my_dict = {"text":eval(x)["text"], "answer_start":eval(x)["answer_start"]}
            aux_list.append(my_dict)
        return {"texts":aux_list}

    def prepare_dataset(self):
        dataset = self.ds.map(self.convert_text, batched=True)
        dataset = dataset.remove_columns("answers")
        dataset = dataset.rename_column("texts", "answers")
        return dataset
