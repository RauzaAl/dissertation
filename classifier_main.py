import argparse

from classes import class_names
from classifier import (
    load_train_dataset, load_test_dataset,
    define_classifier,
    Classifier,
    plot_metrics,
)


parser = argparse.ArgumentParser(description='Classifier model')
parser.add_argument('--action', type=str, required=True, help='Action to perform')
parser.add_argument('--train-dir', type=str, help='Train data directory')
parser.add_argument('--test-dir', type=str, help='Test data directory')
parser.add_argument('--out-model', type=str, help='Path to save trained model')
parser.add_argument('--report-file', type=str, help='Path to report file with metrics of accuracies and losses')
parser.add_argument('--model-path', type=str, help='Path to saved model to load')
parser.add_argument('--image-path', type=str, help='Path to image to test')
parser.add_argument('--output-file', type=str, help='Path of output file')


def main(args):
    action = args.action

    if action == 'train_and_test':
        train_dir = args.train_dir
        test_dir = args.test_dir
        out_model = args.out_model
        report_file = args.report_file

        train_and_test(train_dir, test_dir, out_model, report_file)

    if action == 'test':
        model_path = args.model_path
        test_dir = args.test_dir

        test(model_path, test_dir)

    if action == 'single_test':
        model_path = args.model_path
        image_path = args.image_path

        single_test(model_path, image_path)

    if action == 'plot_metrics':
        report_file = args.report_file
        output_file = args.output_file

        plot_metrics(report_file, output_file)


def train_and_test(train_dir: str, test_dir: str, out_model: str, report_file: str):
    model = Classifier()

    X_train, y_train = load_train_dataset(train_dir)
    model.set_train_data(X_train, y_train)

    cl = define_classifier()
    model.set_model(cl)

    model.train(out_model)
    model.import_metrics(report_file)

    X_test, y_test = load_test_dataset(test_dir)
    model.set_test_data(X_test, y_test)

    test_acc = model.test()
    print(f'Test accuracy: {test_acc}')


def test(model_path: str, test_dir: str):
    model = Classifier()

    cl = define_classifier()
    model.set_model(cl)
    model.load_model(model_path)

    X_test, y_test = load_test_dataset(test_dir)
    model.set_test_data(X_test, y_test)

    test_acc = model.test()
    print(f'Test accuracy: {test_acc}')


def single_test(model_path: str, image_path: str):
    model = Classifier()

    cl = define_classifier()
    model.set_model(cl)
    model.load_model(model_path)

    predicted_class = model.test_one(image_path)
    class_name = class_names[predicted_class]

    print(f'Predicted image class: {class_name} (class_label: {predicted_class})')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
