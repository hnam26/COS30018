import os
import json
from datasets import Dataset, DatasetDict
import pandas as pd
class DataPreprocessor:
    def __init__(self, data_files, data_folder='data'):
        self.data = {}
        for key, files in data_files.items():
            self.data[key] = []
            if not isinstance(files, list):
                files = [files]
            for file in files:
                file_path = os.path.join(data_folder, file)  # prepend the folder path
                with open(file_path, 'r') as f:
                    self.data[key].extend(json.load(f)['data'])

    def preprocess(self):
        preprocessed_data = {}
        for key, data in self.data.items():
            preprocessed_data[key] = {
                'id': [],
                'context': [],
                'question': [],
                'answers': []
            }
            for topic in data:
                for paragraph in topic['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        id = qa['id']
                        for answer in qa['answers']:
                            preprocessed_data[key]['id'].append(id)
                            preprocessed_data[key]['context'].append(context)
                            preprocessed_data[key]['question'].append(question)
                            preprocessed_data[key]['answers'].append({
                                'answer_start': [answer['answer_start']],
                                'text': answer['text']
                            })
        return preprocessed_data

    def prepare_dataset(self):
        preprocessed_data = self.preprocess()
        datasets = DatasetDict()
        for key, data in preprocessed_data.items():
                df = pd.DataFrame(data)
                df['id'] = df['id'].astype(str)  # convert 'id' to string
                datasets[key] = Dataset.from_pandas(df)
        return datasets
    