from transformers import TrainingArguments, pipeline, BertTokenizer, Trainer
import torch
from utils.regressor import *

base_model = "bert-base-multilingual-cased"
local_model = "U-NON-GEO+GEO-ONLY-O1-d-total_mean-mf_mean-pos_spher-N30e5-B10-E3-cosine-LR[1e-05;1e-06].pth"
hub_model = "k4tel/bert-geolocation-prediction"

if torch.cuda.is_available():
    state = torch.load(local_model)
    device = torch.device("cuda")
    state = torch.load(local_model)
else:
    state = torch.load(local_model, map_location='cpu')
    device = torch.device("cpu")
    state = torch.load(local_model, map_location='cpu')

# AutoConfig.register("bert-geo-regressor", BertRegressorConfig)
# AutoModel.register(BertRegressorConfig, BertGeoRegressor)

# PIPELINE_REGISTRY.register_pipeline(
#     "geo-regression",
#     pipeline_class=RegressionPipeline,
#     pt_model=BertGeoRegressor,
# )

# BertRegressorConfig.register_for_auto_class()
# BertRegressor.register_for_auto_class("AutoModel")
# BertForMaskedLM.register_for_auto_class("AutoModelForSequenceClassification")

#wrapper = BERTregModel(5, "spher", True, ["NON-GEO", "GEO-ONLY"], base_model)



# bert = BertForMaskedLM(config)
# bert.base_model.load_state_dict(state['model_state_dict'], strict=False)
# bert.push_to_hub("k4tel/bert-multilingial-geolocation-prediction")

# pipe = pipeline("text-classification")
# res = pipe("Hello I'm from Berlin")
# print(res)

# config = AutoConfig.from_pretrained(base_model)
# config.push_to_hub(hub_model)

config = BertRegressorConfig().from_pretrained(pretrained_model_name_or_path=hub_model, force_download=True)
tokenizer = BertTokenizer.from_pretrained(base_model)


model = BertGeoRegressor(config).to(device)
model.load_state_dict(state_dict=state['model_state_dict'], strict=False)

args = TrainingArguments(
    hub_model,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    push_to_hub=True
)

trainer = Trainer(
    model,
    args,
    tokenizer=tokenizer
)


trainer.create_model_card(tasks=["feature-extraction", "sentiment-analysis"],
                          finetuned_from=base_model)
#trainer.save_model(hub_model)

classifier = pipeline("feature-extraction", model=hub_model)
classifier.save_pretrained(hub_model)

trainer.push_to_hub(commit_message="fe")

model = BertModel.from_pretrained(pretrained_model_name_or_path=base_model)
print(model)

model = BertModel.from_pretrained(pretrained_model_name_or_path=hub_model, config=config)
print(model)
# classifier = pipeline("feature-extraction",  device=device, model=model, tokenizer=tokenizer)
# res = classifier("Hello I'm from Berlin")
# print(res)



