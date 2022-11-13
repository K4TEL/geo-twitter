from utils.result_visuals import *
from utils.regressor import *
import torch

# results manager and visual test on evaluated datasets

feature = "full"
file = f"pmop-test"

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
model = ModelBenchmark(bert_wrapper, distance=True, loss_prob="pos", mf_loss="sum", total_loss="sum")
result = ResultManager(None, feature, device, model, scaled=False, prefix=file)
result.load_df(input_pred)


#visual = ResultVisuals(result)
#visual.gaus_compare()
#visual.prob_map_animation(228)
#visual.gaus_map()
#visual.density()
#visual.cum_dist(False, 200)
#visual.points_on_map(output_map_point)
#visual.interact_lines(10)
#visual.save_df()

result.result_metrics(True, 100)
result.result_metrics(False, 100)



