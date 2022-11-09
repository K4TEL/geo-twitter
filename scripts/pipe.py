import numpy as np
import torch
import torch.nn as nn

from transformers import Pipeline

class RegressionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        encoded_corpus = self.tokenizer(text=inputs,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        truncation='longest_first',
                                        max_length=300,
                                        return_attention_mask=True)
        return {"model_input": encoded_corpus}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(torch.tensor(model_inputs['model_input']['input_ids']).reshape(1, -1).to(torch.int64),
                             torch.tensor(model_inputs['model_input']['attention_mask']).reshape(1, -1).to(torch.int64))
        return outputs.numpy()

    def postprocess(self, model_outputs):
        return model_outputs
