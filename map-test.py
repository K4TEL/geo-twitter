from utils.result_visuals import *
from utils.regressor import *
import torch

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

# dims = [[0, 0, 1],
#         [0, 1, 0],
#         [1, 0, 0]]
# angles = [0, 0, 0]
# step = 15
# for angle_step_z in range(13):
#     angles[0] = angle_step_z*step
#     for angle_step_y in range(13):
#         angles[1] = angle_step_y*step
#         for angle_step_x in range(1, 13):
#             angles[2] = angle_step_x*step
#             filename = f"hpcaori_rotated_z{angles[0]}y{angles[1]}x{angles[2]}"
#
#             rotating_object = original_object
#             print(filename)
#             for rotate_coord in range(3):
#                 current_dim = dims[rotate_coord]
#                 current_angle = angles[rotate_coord]
#
#                 rotating_object = md.transformations.rotate.rotateby(current_angle, direction=current_dim, ag = prettyclose)
#
#             modelobject = rotating_object
#
#             ElizabethII= md.Merge(modelobject, u.select_atoms('segid MEMB'),
#                      u.select_atoms('segid WAT'), u.select_atoms('segid IONS'))
#             CharlesIII = ElizabethII.atoms.write(f'prettyclose_{current_angle}{current_dim}.pdb')
#             bilbo = { 0: 'Z', 1: 'Y', 2: 'X' }
#             axes = bilbo[int(coord)]
#             md.Universe(CharlesIII)
#             CharlesIII.coordinates.GRO.GROWriter(f'prettyclose_{current_angle}{axes}.gro', reindex = True)
#             CharlesIII.coordinates.GRO.GROWriter.fmt['box_orthorhombic'].format(box=(134.618, 134.618, 225.007))



