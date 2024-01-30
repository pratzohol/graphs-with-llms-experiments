import argparse
from datetime import date

def get_params():
    args = argparse.ArgumentParser()

    args.add_argument("--exp_name", default="evaluation-mode", type=str) # For easy identification of different experiments
    args.add_argument("--dataRoot", default="./data", type=str)  # path to original data
    args.add_argument("--custom_dataRoot", default="./custom_data", type=str)  # path to processed custom data

    args.add_argument("--dataset", default="arxiv", type=str) # dataset name
    args.add_argument("--sentence_encoder", default="ST", type=str) # type of sentence encoder
    args.add_argument("--model_type", default="MLP", type=str) # type of model to be used, e.g., mlp, gcn, gat, sage

    args.add_argument("--state_dict_path", default="./state_dicts", type=str) # path to store state_dicts of trained models

    # Training-specific params
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--epochs", default=200, type=int)
    args.add_argument("--batch_count", default=501, type=int) # number of batches
    args.add_argument("--batch_size", default=5, type=int) # size of each batch
    args.add_argument("--weight_decay", default=0.001, type=float)
    args.add_argument("--dropout", default=0.3, type=float)
    args.add_argument("--seed", default=None, type=int)


    args.add_argument("--workers", default=10, type=int) # Number of workers per dataloader
    args.add_argument("--device", default=123, type=int) # device 123 means CPU

    args.add_argument("--eval_only", default=False, type=bool) # If True, only evaluates the model

    ################################################################################
    args = args.parse_args()

    params = vars(args)
    params['device'] = 'cpu' if args.device==123 else f"cuda:{args.device}"
    params["exp_name"] = f"Date -> {date.today()}. Experiment_{args.sentence_encoder}_{args.exp_name}"

    return params

if __name__ == "__main__":

    params = get_params()
    print(params)