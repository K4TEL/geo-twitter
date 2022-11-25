from utils.result_visuals import *
from utils.regressor import *
import torch

# results manager and visual test on evaluated datasets

feature = "NON-GEO"
file = f"U-NON-GEO+GEO-ONLY-O5-d-total_mean-mf_mean-pos_spher-weighted-N30e5-B10-E3-cosine-LR[1e-05;1e-06]_predicted_N1000_VF-NON-GEO_2022-11-25"

input_pred = f"results/val-data/{file}.jsonl"

# output_pred = f"results/val-data/{file}-out.jsonl"
# output_map_point = f"img/map-test-{feature}.png"
# output_map_line = f"img/map-test-{feature}.png"
# output_dist = f"img/dist-test-{feature}.png"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Available GPU has {torch.cuda.device_count()} devices, using {torch.cuda.get_device_name(0)}")
else:
    print(f"No GPU available, using the CPU with {torch.get_num_threads()} threads instead.")
    device = torch.device("cpu")

bert_wrapper = BERTregModel(n_outcomes=5, covariance="spher", weighted=True, features=["NON-GEO", "GEO-ONLY"])
model = ModelBenchmark(bert_wrapper, distance=True, loss_prob="pos", mf_loss="mean", total_loss="mean")
result = ResultManager(None, None, feature, device, model, scaled=False, by_user=False, prefix=file)
result.load_df(input_pred)

# metrics
# result.result_metrics(True, 100)
# result.result_metrics(False, 100)

# result.performance()

visual = ResultVisuals(result)

# standard
# visual.density()
# visual.cum_dist(False, 100)

# GMM
# visual.summarize_prediction(1)
# visual.gaus_map()
# visual.prob_map_animation(228)

visual.interactive_map(lines=False, best=True)

# result.save_df()



