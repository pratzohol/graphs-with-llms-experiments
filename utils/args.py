import argparse
from datetime import date

def get_params():
    args = argparse.ArgumentParser()

    args.add_argument("--keyword", default="Evaluation", type=str) # For easy identification of different experiments
    args.add_argument("--dataRoot", default="./data", type=str)  # path to original data
    args.add_argument("--custom_DataRoot", default="./custom_data", type=str)  # path to processed custom data

    args.add_argument("--dataset", default="arxiv", type=str) # dataset name
    args.add_argument("--sentence_encoder", default="ST", type=str) # type of sentence encoder

    # early stopping patience (in validation epochs, so with default eval_epoch argument 20 * 10 = 200 epochs)
    args.add_argument("--early_stopping_patience",default=20, type=int)

    args.add_argument("--seed", default=None, type=int)

    # Training-specific params
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--epochs", default=12, type=int)
    args.add_argument("--batch_size", default=5, type=int)
    args.add_argument("--weight_decay", default=0.001, type=float)
    args.add_argument("--dropout", default=0, type=float)

    # Number of workers per dataloader # default=10 <--- original
    args.add_argument("--workers", default=10, type=int)
    args.add_argument("--device", default=123, type=int)  # device 123 means CPU

    args.add_argument("--eval_only", default=False, type=bool)

    args.add_argument("--log_dir", default="log", type=str)
    args.add_argument("--state_dir", default="state", type=str)


    ################################################################################
    args = args.parse_args()

    params = {}
    for k, v in vars(args).items():
        params[k] = v

    params['device'] = 'cpu' if args.device==123 else f"cuda:{args.device}"
    params["exp_name"] = f"Date -> {date.today()}. Experiment_{args.sentence_encoder}_{args.keyword}"

    return params
