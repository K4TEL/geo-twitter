
# Geolocation Prediction BERT model

This project is aimed to solve the tweet/user geolocation prediction task and provide a flexible methodology for the geotagging of textual big data. The suggested approach implements neural networks for natural language processing (NLP) to estimate the location as coordinates (longitude, latitude) and two-dimensional Gaussian Mixture Models (GMMs). The scope of proposed models has been finetuned on a Twitter dataset using pretrained Bidirectional Encoder Representations from Transformers (BERT) as a base model. 

[Predicting the Geolocation of Tweets Using BERT-Based Models Trained on Customized Data](https://arxiv.org/pdf/2303.07865.pdf) - paper pre-print on arXiv 

[geo-bert-multilingual](https://huggingface.co/k4tel/geo-bert-multilingual) - repository on HuggingFace of the best model (Probabilistic, 5 outcomes, NON-GEO + GEO-ONLY) trained on the worldwide Twitter dataset

## Project structure

- **datasets** - source folder for the input dataset files used during training and evaluation. For correct reading, the format of the files should be .jsonl containing "lon", "lat", "text", "user" and "place" columns (JSON object fields).

- **models** - folder containing files of local models and checkpoints in .pth format.

- **results** - folder for output files such as images, evaluated datasets, and performance metric reports. 

- **utils** - folder containing vital utility python classes
    - `benchmark.py` - loss function computation and Tensorboard log of training metrics
    - `scheduler.py` - [Cyclic Cosine Decay Learning Rate Scheduler](https://github.com/abhuse/cyclic-cosine-decay)
    - `twitter_dataset.py` - dataset wrapper class implements features forming, tokenization, and creation of PyTorch dataloaders
    - `regressor.py` - linear regression wrapper layer for BERT base models
    - `result_manager.py` - postprocessing of model outputs, writing and reading of evaluation results .jsonl files, performance metrics computation
    - `result_visuals.py` - visualization of results on matplotlib plots 
    - `prediction.py` - single text prediction routine
    - `model_trainer.py` - training and evaluation of the models

- `train_bert.py` - command line parameters input, entry point for training and evaluation
- `input_entry.py` - entry point for single text prediction using local or HF repository models

Additional:

- **runs** - folder for storing training Tensorboard log files 

- **scripts appendix** - folder containing testing and development python scripts, and bash scripts for running jobs on a cluster with slurm management system 

- `valid_data.py` - shortcut for results management and visualization
- `collector.py` - parsing of Twitter database files to collect dataset files

## Usage/Examples

To run the project locally you can clone this project with:

```bash
git clone https://github.com/K4TEL/geo-twitter.git
```

Then, in your python environment run:

```bash
pip install -r requirements.txt
```

### Training

**NOTE!** To run finetuning training place dataset file (.jsonl) containing "lon", "lat", "text", "user" and "place" columns (JSON object fields, no headers required) into the **datasets** folder. 
Then change the dataset file name in `train_bert.py` manually or by passing `-d <dataset_filename>.jsonl` argument. 

To launch the finetuning training with default hyperparameters run:

```bash
  python train_bert.py --train
```

You can change default hyperparameters manually in `train_bert.py` or pass command line arguments by using predefined flags. 
The list of all flags could be found in the same entry point file.

In practice, learning rate, scheduler type, number of epochs, loss function parameters and target columns should remain the same. 
Commonly changeable parameters include number of outcomes, covariance type, features, dataset file name, training dataloader size, batch size and log step.

During finetuning, training metrics and test metrics (calculated at the end of each epoch) are written to the **runs** folder.
The tracking of models performance is implemented using the Tensorboard python library.
Model files and their checkpoints are saved to the **models** directory automatically.  

### Evaluation

**NOTE!** To run evaluation place a dataset file into the **datasets** folder. 
And make sure you have a file of the finetuned model in .pth format in the **models** directory.

To launch the evaluation with default settings run:

```bash
  python train_bert.py --eval
```

In this case, the model file would be chosen automatically according to the file name prefix formed from the preset hyperparameters. 
To pick the model manually you should adjust hyperparameters (number of outcomes, covariance type, features, loss function type) to match the previously finetuned model and run:

```bash
  python train_bert.py --eval -m <model_filename>
```

Commonly changeable parameters for the evaluation are dataset file name, validation dataloader size and model file name.

To perform per user evaluation use `-vu -v <N>` flags that will pick N users with the highest number of samples from the dataset. 
In this case, performance metrics computation takes average per user values rather than average per tweet. 
Note that only probabilistic models using GMMs could summarize multiple per tweet predictions.  

The results of evaluation are written to the .jsonl dataset file containing input and output of the model. 
By default, performance metrics are calculated in the end and written to a short report file of .txt format. 
The visualization of error distance density and its cumulative distribution per outcome are drawn to .png files.

Using `valid_map.py` you can read saved predictions files and use visualization functions more easily.

All outputs of the evaluation are stored in the **results** folder.

### Prediction

**NOTE!** To run single text prediction you should place .pth finetuned model files in the **models/final** directory.

To launch the prediction with default settings run:

```bash
    python input_entry.py
```
Parameters like number of outcomes, probabilistic or geospatial model type, local model file and text could be specified by flags:

```bash
    python input_entry.py -m <model_filename> -t <text>
```

## Support

For support, email lutsai.k@gmail.com
