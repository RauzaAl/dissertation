# Traffix

## Setup

Create and setup virtualenv
```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Unzip artifacts from `output.zip` archive.

`PREPARE.md` file contains information about setting up machine with GPU.

## Structure

1. `output.zip` contains directory with artifacts of both models: classifier and gan. Each directory contains checkpoints for loading model, csv file with metrics from training process and visualisation of that metrics on image. There are 3 classification artifacts for real, synthetic and merged datasets.
2. `gan` directory contains high-level functions and classes for conditional GAN model.
3. `classification` directory contains high-level functions and classes for classification model.
4. `data.py` contains low-level functions for loading datasets.
5. `classes.py` contains human readable names of classes from dataset.
6. `*_main.py` scripts are entrypoints for executing scripts.

## GAN

Train new gan model
```shell
python gan_main.py --action="train" --data-dir="./data" --cp-dir="./gan-cp" --out-dir="./gan-train" --report-file="./losses.csv"
```

Generate images for each class from checkpoint
```shell
python gan_main.py --action="generate" --cp-path="./gan-cp/1" --out-dir="./gan-generate" --count=1000
```

Plot loss metrics from file
```shell
python gan_main.py --action="plot_metrics" --report-file="./losses.csv" --output-file1="./losses_by_batches.png" --output-file2="./losses_by_epochs.png"
```

## Classifier

Train and test
```shell
python classifier_main.py --action="train_and_test" --train-dir="./data/Train" --test-dir="./data" --out-model="./model.h5" --report-file="./metrics.csv"
```

Separately test model on whole test dataset
```shell
python classifier_main.py --action="train_and_test" --model-path="./model.h5" --test-dir="./data"
```

Make test for single image
```shell
python classifier_main.py --action="single_test" --model-path="./model.h5" --image-path="./data/Test/1.png"
```

Plot accuracy and loss metrics from file
```shell
python classifier_main.py --action="plot_metrics" --report-file="./metrics.csv" --output-file="./metrics.png"
```
