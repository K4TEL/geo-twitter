
# Geolocation Prediction BERT model

This project is aimed to solve the tweet/user geolocation prediction task and provide a flexible methodology for the geotagging of textual big data. The suggested approach implements neural networks for natural language processing (NLP) to estimate the location as coordinates (longitude, latitude) and two-dimensional Gaussian Mixture Models (GMMs). The scope of proposed models has been finetuned on a Twitter dataset using pretrained Bidirectional Encoder Representations from Transformers (BERT) as a base model. 

## Project structure

The project structure reduced to classes and function required to run a single-text prediction using HF repo model and process the outputs.

- `text_result.py` - loading model from the HF repo and post-processing outputs of a single text prediction 
- `prediction.ipynb` - Jupyter Notebook to load and perform prediction

## Usage/Examples

To run the project locally you can clone this project with:

```bash
git clone -b predict https://github.com/K4TEL/geo-twitter.git
```

Then, in your python environment run:

```bash
pip install -r requirements.txt
```

### Prediction

**NOTE!** To run single text prediction HF repository model is loaded.

To launch the prediction with default settings run:

```bash
    python text_result.py
```

The setup is ought to be changed manually by changing the content of `main()` function in the script. There are options to use a custom HF pipeline or manually process inputs and outputs. 

### Google colab example

[Google Colab notebook of the predication demo](https://colab.research.google.com/drive/1Nu9VMlPpsATxCyEge-sBBOwFZcqKHJfM?usp=sharing) is a copy of the `prediction.ipynb` and available to launch.

## Support

For support, email lutsai.k@gmail.com
