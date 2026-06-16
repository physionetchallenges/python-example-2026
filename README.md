# Python example code for the George B. Moody PhysioNet Challenge 2026

## What's in this repository? 

This repository contains a simple example that illustrates how to format a Python entry for the [George B. Moody PhysioNet Challenge 2026](https://physionetchallenges.org/2026/). If you are participating in the 2026 Challenge, then we recommend using this repository as a template for your entry. You can remove some of the code, reuse other code, and add new code to create your entry. You do not need to use the models, features, and/or libraries in this example for your entry. We encourage a diversity of approaches to the Challenges.

For this example, we implemented a random forest model with several simple features. (This simple example is **not** designed to perform well, so you should **not** use it as a baseline for your approach's performance.) You can try it by running the following commands on the Challenge training set. If you are using a relatively recent personal computer, then you should be able to run these commands from start to finish on a small subset (1000 records) of the training data in a few minutes or less.

## How do I run these scripts?

First, you can download and create data for these scripts by following the [instructions](https://github.com/physionetchallenges/python-example-2026?tab=readme-ov-file#how-do-i-create-data-for-these-scripts) in the following section.

Second, you can install the dependencies for these scripts by creating a Docker image (see below) or [virtual environment](https://docs.python.org/3/library/venv.html) and running

    pip install -r requirements.txt

You can train your model by running

    python train_model.py -d training_data -m model

where

- `training_data` (input; required) is a folder with the training data files, which must include the labels; and
- `model` (output; required) is a folder for saving your model.

You can run your trained model by running

    python run_model.py -d holdout_data -m model -o holdout_outputs

where

- `holdout_data` (input; required) is a folder with the held-out data files, which will not necessarily include the labels;
- `model` (input; required) is a folder for loading your model; and
- `holdout_outputs` (output; required) is a folder for saving your model outputs for the held-out data.

The [Challenge website](https://physionetchallenges.org/2026/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2026) and running

    python evaluate_model.py -d holdout_labels -o holdout_outputs -p prevalence_labels -s scores.csv

where

- `holdout_labels`(input; required) is the path to the CSV file with the labels for the held-out data;
- `holdout_outputs` (input; required) is the path to the CSV file with your model's outputs for the held-out data;
- `prevalence_labels` (input; required) is the path to the CSV file with labels and ages for the data; we will use the `demographics.csv` data for the full training set;
- `scores.tsv` (output; optional) is file with a collection of scores for your model.

You can use the provided training set for the `training_data` and `holdout_data` files, but we will use different datasets for the validation and test sets, and we will not provide the labels to your code.

## How do I create data for these scripts?

Please see the [data](https://physionetchallenges.org/2026/#data) section of the website for more information about the Challenge data.

## Which scripts I can edit?

Please edit the following script to add your code:

* `team_code.py` is a script with functions for training and running your trained model.

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model.
* `run_model.py` is a script for running your trained model.
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

To train and save your model, please edit the `train_model` function in the `team_code.py` script. Please do not edit the input or output arguments of this function.

To load and run your trained model, please edit the `load_model` and `run_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of these functions.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 1000 records.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data holdout_data model holdout_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2026/#data). Put some of the training data in `training_data` and `holdout_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2026.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        holdout_data  holdout_outputs  model  python-example-2026  training_data

        user@computer:~/example$ cd python-example-2026/

        user@computer:~/example/python-example-2026$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2026$ docker run -it -v ~/example/model:/challenge/model -v ~/example/holdout_data:/challenge/holdout_data -v ~/example/holdout_outputs:/challenge/holdout_outputs -v ~/example/training_data:/challenge/training_data image bash

        root@[...]:/challenge# ls
            Dockerfile             holdout_outputs        run_model.py
            evaluate_model.py      LICENSE                training_data
            helper_code.py         README.md      
            holdout_data           requirements.txt

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d holdout_data -m model -o holdout_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d holdout_data -o holdout_outputs
        [...]

        root@[...]:/challenge# exit
        Exit

## What else do I need?

Please see the [evaluation code repository](https://github.com/physionetchallenges/evaluation-2026) for code and instructions for evaluating your entry using the Challenge scoring metric.

## How do I learn more? How do I share more?

Please see the [Challenge website](https://physionetchallenges.org/2026/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges). Please do not make pull requests, which may share information about your approach.

## Useful links

* [Challenge website](https://physionetchallenges.org/2026/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2026)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2026)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2026/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
