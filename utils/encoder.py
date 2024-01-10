import os
import os.path as osp
import torch
from sentence_transformers import SentenceTransformer
from transformers import LlamaTokenizer, LlamaModel


class SentenceEncoder:
    def __init__(self, name, root="lang_models", batch_size=512, device=0, multi_gpu=False):
        self.name = name
        self.root = osp.abspath(root)
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu

        # if device = 123 then use cpu otherwise use cuda
        self.device = device #"cpu" if device==123 else f"cuda:{device}"

        if self.name == "ST":
            self.model = SentenceTransformer(
                osp.join(self.root, "sentence-transformers_multi-qa-distilbert-cos-v1"),
                device=self.device,
                cache_folder=root
            )

        elif self.name == "llama2":
            model_path = osp.join(self.root, "llama-2-7b")
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path, device=self.device)
            self.model = LlamaModel.from_pretrained(model_path).to(self.device)

        elif self.name == "roberta":
            self.model = SentenceTransformer(
                osp.join(self.root, "sentence-transformers_roberta-base-nli-stsb-mean-tokens"),
                device=self.device,
                cache_folder=root
            )
        else:
            raise ValueError(f"Unknown language model: {name}.")

    def encode(self, texts):
        if self.multi_gpu:
            # Start the multi-process pool on all available CUDA devices
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process( texts, pool=pool, batch_size=self.batch_size)
            embeddings = torch.from_numpy(embeddings)
        else:
            # return tensor instead of list of python integers
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=True)

        # returns to self.device
        return embeddings


