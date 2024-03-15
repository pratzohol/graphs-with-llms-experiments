import torch
import torch.nn as nn

class GetLossAcc:
    def __init__(self, model_option, params):
        self.model_option = model_option

        self.n_way = params["n_way"]

    def get_functions(self):
        if self.model_option == 1:
            return self.loss_1, self.accuracy_1
        elif self.model_option == 2:
            return self.loss_2, self.accuracy_2
        else:
            raise ValueError(f"Unknown model option: {self.model_option}")

    def loss_1(self):
        pass

    def loss_2(self, class_emb, correct_label_mask):
        loss = nn.BCEWithLogitsLoss(reduction="sum")
        return loss(class_emb.squeeze(), correct_label_mask.type(torch.float)) / self.n_way

    def accuracy_1():
        pass

    def accuracy_2(self, class_emb, correct_label_mask):
        sigmoid = nn.Sigmoid()
        prob = sigmoid(class_emb)

        ypred = prob.reshape(-1, self.n_way)
        ytrue = correct_label_mask.type(torch.float).reshape(-1, self.n_way)

        # print("ypred:", ypred)
        # print("ytrue:", ytrue)

        pred = ypred.argmax(dim=1)
        true = ytrue.argmax(dim=1)

        total_count = true.shape[0]

        return (pred == true).sum(), total_count # gives total number of correct predictions and total number of predictions