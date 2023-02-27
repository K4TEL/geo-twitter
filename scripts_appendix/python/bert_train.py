import argparse
from transformers import BertTokenizer
from utils.model_trainer import *

# Entry point for training anf evaluation of the models

f = ["GEO", "NON-GEO", "META-DATA", "TEXT-ONLY", "GEO-ONLY", "USER-ONLY"]

dataset_file = 'worldwide-twitter-day-small.jsonl'  # .jsonl
features = [f[1], f[4]]
val_f = None  # None -> features[0]
target_columns = ["lon", "lat"]

original_worldwide_model = "bert-base-multilingual-cased"
original_usa_model = "bert-base-cased"

# parameters = dict(
#     max_lr = [5e-5, 1e-5],
#     min_lr = [5e-6, 1e-6, 1e-8, 1e-16],
#     scheduler = ["cosine", "plateau"]
# )
# param_values = [v for v in parameters.values()]

covariance_types = [None, "spher"]  # [None, "full", "spher", "diag", "tied"]
scheduler_types = ["cosine", "linear", "plateau"]  # ["cosine", "linear", "cosine-long", "plateau", "step", "multi step", "one cycle", "cyclic"]

loss_distance = True
loss_mf = "mean"  # mean/sum - mean if features > 1
loss_prob = "pos"  # all/pos - pos if prob
loss_total = "mean"  # sum/mean/type - mean if prob else type (spat)

outcomes = 5
covariance = covariance_types[1]  # None/spher

epochs = 3
log_step = 1000

batch_size = 4

lr_max = 1e-5
lr_min = 1e-6
scheduler = scheduler_types[0]

val_size = 1000  # samples/users if -vu
threshold = 100

train_size = 0
test_ratio = 0.1
seed = 42


def main():
    parser = argparse.ArgumentParser(description='Finetune multilingual transformer model')

    # common
    parser.add_argument('-n', '--n-epochs', type=int, default=epochs, help='Number of the training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=batch_size, help='Batch size (default: 12)')

    parser.add_argument('-m', '--local_model', type=str, default=None,
                        help='Filename prefix of local model (!NOTE must fit related args)')

    parser.add_argument('-bb', '--bert_base', type=str, default=None,
                        help="BERT base model (default: worldwide)")


    parser.add_argument('--train', action="store_true", help="Start finetuning")
    parser.add_argument('--eval', action="store_true", help="Start evaluation")
    parser.add_argument('--hptune', action="store_true", help="Start training with hyper parameters tuning")

    # dataset
    parser.add_argument('-d', '--dataset', type=str, default=dataset_file, help="Input dataset (in jsonl format)")
    parser.add_argument('-ss', '--skip', type=int, default=0, help='Number of dataset samples to skip')

    parser.add_argument('-f', '--features', default=features, nargs='+', help="Feature column names")
    parser.add_argument('-t', '--target-cols', default=target_columns, nargs='+', help="Target column names")

    parser.add_argument('-s', '--seed', type=int, default=seed, help='Random seed (default: 42)')


    parser.add_argument('-nnv', '--norm_numb', action="store_true",
                        help="Normalize labels' number values (default: False)")

    parser.add_argument('-cv', '--class_val', action="store_true",
                        help="Classification labels as values (default: False)")


    # --train
    parser.add_argument('--no-ckp', action="store_false", help='Saving model checkpoints during training (Default: True)')

    parser.add_argument('-ls', '--log_step', type=int, default=log_step, help='Log step (default: 1000)')

    parser.add_argument('-lr', '--learn_rate', type=float, default=lr_max,
                        help='Learning rate maximum to start from (default: 4e-5)')
    parser.add_argument('-lrm', '--learn_rate_min', type=float, default=lr_min,
                        help='Learning rate minimum to end on (default: 1e-8)')
    parser.add_argument('-sdl', '--scheduler', type=str, default=scheduler, help="Scheduler type (Default: 'cosine')")

    parser.add_argument('-ts', '--train_size', type=int, default=train_size, help='Training dataloader size')
    parser.add_argument('-tr', '--test_ratio', type=float, default=test_ratio, help='Training dataloader test data ratio (default: 0.1)')

    parser.add_argument('-lmf', '--loss_mf', type=str, default=loss_mf,
                        help="Multi feature loss handle mean or sum (default: mean)")


    # --eval
    parser.add_argument('-v', '--val_size', type=int, default=val_size, help='Validation dataloader size')

    # geo specific
    parser.add_argument('-o', '--outcomes', type=int, default=outcomes, help="Number of outcomes (lomg, lat) per tweet")
    parser.add_argument('-ld', '--loss_dist', action="store_false", help="Distance loss criterion (default: True)")

    parser.add_argument('-lp', '--loss_prob', type=str, default=loss_prob,
                        help="GMM neg LLH loss stays in domain 'all' or 'pos' (default: 'pos')")
    parser.add_argument('-c', '--covariance', type=str, default=covariance,
                        help="GMM covariance matrix type (Default: 'spher')")
    parser.add_argument('-nw', '--not-weighted', action="store_false",
                        help="GMM weights are not equal (default: True)")

    parser.add_argument('-lt', '--loss_total', type=str, default=loss_total,
                        help="Total loss handle by model's 'type' or 'sum' of all loss values (default: 'type')")

    args = parser.parse_args()

    if args.local_model is None:
        prefix = f"{'US-' if args.usa_model else ''}{'U-' if not args.scale_coord else ''}{'+'.join(args.features)}-O{args.outcomes}-{'d' if  args.loss_dist else 'c'}-" \
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

    original_model = original_usa_model if args.usa_model else original_worldwide_model

    dataloader = TwitterDataloader(args.dataset,
                                   args.features,
                                   target_columns,
                                   BertTokenizer.from_pretrained(original_model),
                                   args.seed,
                                   args.scale_coord,
                                   val_f)

# no settings run to save filtered by condition dataset copy
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

    # if args.hptune:
    #     trainer.hp_tuning(args.train_size,
    #                       args.test_ratio,
    #                       param_values,
    #                       args.log_step)

    if args.train:
        trainer.finetune(args.train_size,
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
                     args.val_user,
                     args.train_size)


if __name__ == "__main__":
    main()
