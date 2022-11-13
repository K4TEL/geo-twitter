import argparse

#from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer

from utils.model_trainer import *

feature_columns = ["full", "all", "texts", "place", "user"]
f = ["GEO", "NON-GEO", "META-DATA", "TEXT-ONLY", "GEO-ONLY"]

dataset_file = 'eisenstein.jsonl'
features = [f[1]]
val_f = None
target_columns = ["lon", "lat"]

original_model = "bert-base-multilingual-cased"

#local_model = f'test-bert-ua-{feature_column}.pth'

#ckp_file = f'ckp-bert-ca-{feature_column}.pth'

# output_pred = f"results/{datetime.today().strftime('%Y-%m-%d')}-prediction-ca-{feature_column}.jsonl"
# output_map = f"img/{datetime.today().strftime('%Y-%m-%d')}-map-ca-{feature_column}.png"
# output_dist = f"img/{datetime.today().strftime('%Y-%m-%d')}-dist-ca-test-{feature_column}.png"

parameters = dict(
    max_lr = [5e-5, 1e-5],
    min_lr = [5e-6, 1e-6, 1e-8, 1e-16],
    scheduler = ["cosine", "plateau"]
)
param_values = [v for v in parameters.values()]

covariance_types = [None, "full", "spher", "diag", "tied"]
scheduler_types = ["cosine", "linear", "cosine-long", "plateau", "step", "multi step", "one cycle", "cyclic"]

loss_distance = True
loss_mf = "mean"  # mean/sum
loss_prob = "pos"  # all/pos
loss_total = "mean"  # sum/mean/type

outcomes = 5
covariance = covariance_types[2]  # None/spher

epochs = 3
log_step = 10

batch_size = 4

lr_max = 1e-5
lr_min = 1e-6
scheduler = scheduler_types[0]

val_size = 100
threshold = 100

train_size = 1000
test_ratio = 0.1
seed = 42

def main():
    parser = argparse.ArgumentParser(description='Finetune multilingual transformer model')
    parser.add_argument('-n', '--nepochs', type=int, default=epochs, help='Number of epochs to train')
    parser.add_argument('-ss', '--skip', type=int, default=0, help='Number of dataset samples to skip')

    parser.add_argument('-sc', '--scale_coord', action="store_true", help="Keep coordinates unscaled (default: True)")
    parser.add_argument('-o', '--outcomes', type=int, default=outcomes, help="Number of outcomes (lomg, lat) per tweet")
    parser.add_argument('-c', '--covariance', type=str, default=covariance, help="Covariance matrix type")
    parser.add_argument('-nw', '--weighted', action="store_false", help="Weights of GMM are not equal (default: True)")

    parser.add_argument('-ld', '--loss_dist', action="store_false", help="Distance loss criterion (default: True)")
    parser.add_argument('-lmf', '--loss_mf', type=str, default=loss_mf, help="Multi feature loss handle mean or sum (default: mean)")
    parser.add_argument('-lp', '--loss_prob', type=str, default=loss_prob, help="Probabilistic loss domain all or pos (default: all)")
    parser.add_argument('-lt', '--loss_total', type=str, default=loss_total, help="Total loss handle by model type or sum (default: type)")

    parser.add_argument('-m', '--local_model', type=str, default=None, help='Filename prefix of local model')
    parser.add_argument('--nockp', action="store_false", help='Saving model checkpoints during training (preset: True)')

    parser.add_argument('-lr', '--learn_rate', type=float, default=lr_max, help='Learning rate (default: 4e-5)')
    parser.add_argument('-lrm', '--learn_rate_min', type=float, default=lr_min, help='Learning rate minimum (default: 1e-8)')
    parser.add_argument('-sdl', '--scheduler', type=str, default=scheduler, help="Scheduler type")

    parser.add_argument('-b', '--batch_size', type=int, default=batch_size, help='Per-device batch size (default: 22)')
    parser.add_argument('-ls', '--log_step', type=int, default=log_step, help='Log step (default: 1000)')

    parser.add_argument('-d', '--dataset', type=str, default=dataset_file, help="Input dataset (in jsonl format)")
    parser.add_argument('-f', '--features', default=features, nargs='+', help="Features names")
    # parser.add_argument('-f', '--feature', type=str, default=features, help="Feature column name")
    parser.add_argument('-ts', '--train_size', type=int, default=train_size, help='Training dataloader size')
    parser.add_argument('-tr', '--test_ratio', type=float, default=test_ratio, help='Training dataloader test ratio (default: 0.1)')
    parser.add_argument('-s', '--seed', type=int, default=seed, help='Random seed (default: 42)')

    parser.add_argument('-vd', '--val_dataset', type=str, default=None, help="Validation dataset (in jsonl format)")
    parser.add_argument('-v', '--val_size', type=int, default=val_size, help='Validation dataloader size')
    parser.add_argument('-th', '--threshold', type=int, default=threshold, help='Validation threshold in km (default: 200)')

    # parser.add_argument('-po', '--pred_output', type=str, default=output_pred, help='Output file for predictions (in jsonl format)')
    # parser.add_argument('-mo', '--map_output', type=str, default=output_map, help='Output file for map plot')
    # parser.add_argument('-do', '--dist_output', type=str, default=output_dist, help='Output file for distribution plot')

    parser.add_argument('-t', '--text', type=str, default=None, help="Predict from text")

    parser.add_argument('--train', action="store_true", help="Start pretraining")
    parser.add_argument('--eval', action="store_true", help="Start evaluation")
    parser.add_argument('--hptune', action="store_true", help="Start training with hyper parameters tuning")
    args = parser.parse_args()

    if args.local_model is None:
        prefix = f"EIS-{'U-' if not args.scale_coord else ''}{'+'.join(args.features)}-O{args.outcomes}-{'d' if  args.loss_dist else 'c'}-" \
                 f"total_{args.loss_total if args.covariance is not None else 'type'}-{'mf_' + args.loss_mf + '-' if len(args.features) > 1 else ''}" \
                 f"{args.loss_prob + '_' if args.covariance is not None else ''}{args.covariance if args.covariance is not None else 'NP'}-" \
                 f"{'weighted-' if args.weighted and args.outcomes > 1 else ''}N{args.train_size//100000}e5-" \
                 f"B{args.batch_size}-E{args.nepochs}-{args.scheduler}-LR[{args.learn_rate};{args.learn_rate_min}]"
    else:
        prefix = args.local_model

    print(f"Model prefix:\t{prefix}")
    if torch.cuda.is_available():
        print(f"DEVICE\tAvailable GPU has {torch.cuda.device_count()} devices, using {torch.cuda.get_device_name(0)}")
        print(f"DEVICE\tCPU has {torch.get_num_threads()} threads")
    else:
        print(f"DEVICE\tNo GPU available, using the CPU with {torch.get_num_threads()} threads instead.")

    dataloader = TwitterDataloader(args.dataset,
                                   args.features,
                                   target_columns,
                                   BertTokenizer.from_pretrained(original_model),
                                   args.seed,
                                   args.scale_coord,
                                   val_f)

#    dataloader.filter_dataset("code", None, ["CA", 'FR', 'GB'])

    trainer = ModelTrainer(prefix,
                           dataloader,
                           args.nepochs,
                           args.batch_size,
                           args.outcomes,
                           args.covariance,
                           args.weighted,
                           args.loss_dist,
                           args.loss_mf,
                           args.loss_prob,
                           args.loss_total,
                           args.learn_rate,
                           args.learn_rate_min,
                           original_model)

    if args.text is not None:
        trainer.predicton(args.text)

    if args.hptune:
        trainer.hp_tuning(args.train_size,
                          args.test_ratio,
                          param_values,
                          args.log_step)

    if args.train:
        trainer.pretrain(args.train_size,
                         args.test_ratio,
                         f"{prefix}.pth",
                         args.nockp,
                         args.log_step,
                         args.scheduler,
                         args.skip)

    if args.eval:
        trainer.eval(args.val_size,
                     args.threshold,
                     args.val_size,
                     args.train_size)
        # if args.val_dataset:
        #     trainer.eval_file_mult(args.val_dataset,
        #                            args.outcomes,
        #                            args.val_size,
        #                            args.threshold,
        #                            map_size=args.val_size)

if __name__ == "__main__":
    main()
