import argparse
from process.TokenProcess import Token
from process.DataPreprocessor import DataPreprocessor
from model.Model import Model
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--train', nargs='+', help='List of paths to the train data files.')
    parser.add_argument('--val', nargs='+', help='List of paths to the validation data files.')
    parser.add_argument('--test', nargs='+', help='List of paths to the test data files.')
    parser.add_argument('--model_id', type=str, default='bert-base-uncased', help='Model ID used to finetune on Huggingface.')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for the trained model.')

    args = parser.parse_args()

    token = Token()
    data_files = {"train": args.train, "validation": args.val}
    test_dataset_names = []
    for test_data in args.test:
        name = os.path.splitext(os.path.basename(test_data))[0]
        test_dataset_names.append(name)
        data_files[name] = test_data

    dataset = DataPreprocessor(data_files).prepare_dataset()

    model = Model(args.model_id, dataset, args.output_dir, test_dataset_names)
    model.train_model()
    model.evaluate()