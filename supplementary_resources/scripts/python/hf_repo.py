import torch
from utils.regressor import *
from utils.result_manager import *
from transformers import Pipeline
from transformers import BertConfig, BertTokenizer

base_model = "bert-base-multilingual-cased"

local_model = "P-NON-GEO+GEO-ONLY-O5.pth"

hub_model = 'k4tel/geo-bert-multilingual'
hf_folder = "models/hf/model"

local_model = f'models/final/{local_model}'


# upload local model and save for the future HF repo upload
def save(local_model, hf_folder, base_model):
    config = BertConfig.from_pretrained(base_model)

    feature_outputs = BERTregModel(5, "spher", True, ["NON-GEO", "GEO-ONLY"], base_model).feature_outputs

    custom_model = GeoBertModel(config=config, feature_outputs=feature_outputs)

    if torch.cuda.is_available():
        state = torch.load(local_model)
    else:
        state = torch.load(local_model, map_location='cpu')

    model_state_dict = state['model_state_dict']

    custom_model.load_state_dict(model_state_dict)

    tokenizer = BertTokenizer.from_pretrained(base_model)

    custom_model.save_pretrained(hf_folder)
    tokenizer.save_pretrained(hf_folder)

    # add all files from hf_folder to the HF repo manually


# huggingface framework load from repo + prediction pipeline test
def load(hub_model, base_model):
    model_wrapper = BERTregModel(5, "spher", True, ["NON-GEO", "GEO-ONLY"], base_model, hub_model)
    benchmark = ModelBenchmark(model_wrapper, True, "pos", "mean", "mean")

    tokenizer = BertTokenizer.from_pretrained(hub_model)
    model = model_wrapper.model

    # testing model
    text = "CIA and FBI can track anyone, and you willingly give the data away"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        prob_model = benchmark.prob_models(outputs)

    print(f"RESULT\tPost-processing raw model outputs: {outputs}")
    result = ResultManager(None, text, "NON-GEO", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), benchmark, False, False, hub_model)
    result.soft_outputs(list([prob_model]))

    ind = np.argwhere(np.round(result.weights[0, :] * 100, 2) > 0)
    significant = result.means[0, ind].reshape(-1, 2)
    weights = result.weights[0, ind].flatten()

    sig_weights = np.round(weights * 100, 2)
    sig_weights = sig_weights[sig_weights > 0]

    print(f"RESULT\t{len(sig_weights)} significant prediction outcome(s):")

    for i in range(len(sig_weights)):
        point = f"lon: {'  lat: '.join(map(str, significant[i]))}"
        print(f"\tOut {i + 1}\t{sig_weights[i]}%\t-\t{point}")


# save(local_model, hf_folder, base_model)
# load(hub_model, base_model)
