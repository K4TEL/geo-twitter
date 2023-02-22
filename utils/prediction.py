from transformers import BertTokenizer
import torch
import numpy as np
from pathlib import Path
from utils.twitter_dataset import *
from utils.result_visuals import *

# single text prediction wrapper
# preprocessing and result visual output

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

        self.result = None
        self.visual = None

    def prediction_output(self, text, filtering=True, visual=False):
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

        self.result = ResultManager(None, text, self.feature, self.device, self.benchmark, False, False, self.prefix)
        self.result.soft_outputs(list([prob_model])) if self.cov else self.result.coord_outputs(output)

        if visual:
            self.visual = ResultVisuals(self.result)
            self.visual.text_map_result()

        return self.result
