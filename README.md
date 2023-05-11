
# Telegram bot of the Geolocation Prediction BERT model

This project is aimed to provide open access for Telegram users to use the developed geolocation prediction model. 
The suggested approach implements neural networks for natural language processing (NLP) to estimate the location as two-dimensional Gaussian Mixture Models (GMMs). 
The model has been finetuned on a worldwide Twitter dataset using pretrained Bidirectional Encoder Representations from Transformers (BERT) as a base model. 

## Project structure

The project structure was reduced to classes and functions required to run a single-text prediction using HF repo model and process the outputs.
In addition, Telegram bot source code and Docker files to run it are added.

- `text_result.py` - loading model from the HF repo and post-processing outputs of a single text prediction 
- `bot_src.py` - Telegram bot source code for processing user requests and logging into DB
- `db.db` - database containing request & response logs
- `buildDocker`- initializing a build of the Docker image
- `Dockerfile`- Docker image building instructions
- `runDocker`- running a Docker container based on the built image
- `runBot`- creating a persistent tmux session to run a Docker container

## Usage/Examples

To run the Telegram bot locally you can clone this project with:

```bash
git clone -b bot https://github.com/K4TEL/geo-twitter.git
```

Make sure that Python 3.8 or later version is installed on your machine. Then, in the project directory run:

```bash
./buildDocker
```

This will initialize the Docker image which will be used a base for the future container. 
Then create a new tmux session by running:

```bash
./runBot
```

Open Telegram and find bot [@geobertbot](https://t.me/geobertbot) to use the geolocation prediction model. 
There are three commands available:

- `/start` - start the bot 
- `/info` - get information about the model
- `/predict` _text_ - get prediction results in text and GMM plot forms

## Support

For support, email lutsai.k@gmail.com or message in Telegram @K4TEL