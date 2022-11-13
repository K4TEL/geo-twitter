from transformers import BertTokenizer
import torch
import numpy as np
from pathlib import Path
from utils.twitter_dataset import *
from utils.regressor import *
from utils.benchmarks import *
from utils.result_manager import *
from utils.result_visuals import *


class ModelOutput():
    def __init__(self, wrapper, model_prefix):
        self.prefix = model_prefix
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = wrapper.model.to(self.device)
        local_model = f"models/final/{self.prefix}.pth"
        print(f"LOAD\tLoading model from {local_model}")
        if not Path(local_model).is_file():
            print(f"LOAD [ERROR] Unable to load local model: file {local_model} does not exist")

        state = torch.load(local_model) if torch.cuda.is_available() else torch.load(local_model, map_location='cpu')
        self.model.load_state_dict(state['model_state_dict'])

        self.outcomes = wrapper.n_outcomes
        self.cov = wrapper.cov
        self.weighted = wrapper.weighted
        self.feature = wrapper.features[0]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.benchmark = ModelBenchmark(wrapper, True, "pos", "mean", "mean" if self.cov else "type")

    def prediction_output(self, text, filtering=True):
        if filtering:
            text = nlp_filtering(text)
            print(f"TEXT\tFiltered text: {text}")

        print("TEXT\tTokenizing text to input IDs and attention masks")
        encoded_corpus = self.tokenizer(text=text,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        truncation='longest_first',
                                        max_length=300,
                                        return_attention_mask=True)
        input_id = encoded_corpus['input_ids']
        attention_mask = encoded_corpus['attention_mask']

        input = torch.tensor(input_id).to(self.device).reshape(1, -1)
        mask = torch.tensor(attention_mask).to(self.device).reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input, mask, self.feature)

            if self.cov:
                prob_model = self.benchmark.prob_models(output)

            output = output.cpu().numpy() if torch.cuda.is_available() else output.numpy()

        result = ResultManager(None, text, self.feature, self.device, self.benchmark, False, self.prefix)
        result.soft_outputs(list([prob_model])) if self.cov else result.coord_outputs(output)

        for i in range(self.outcomes):
            weight = np.round(result.weights[0, i] * 100, 2)
            point = f"lon: {'  lat: '.join(map(str, result.means[0, i])) }"
            if weight > 0:
                print(f"\tOut {i+1}\t{weight}%\t-\t{point}")

        visual = ResultVisuals(result)
        visual.text_map_result()
