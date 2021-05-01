import argparse
import functools
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import math
import torch
from torch import nn
import texar.torch as tx
from data_parallel import MyDataParallel
from texar.torch.run import make_deterministic # Make experiment deterministic by using specific random seeds

from model import Transformer
import utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-model", type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    "--config-data", type=str, default="config_data",
    help="The dataset config.")
parser.add_argument(
    "--run-mode", type=str, default="train",
    help="Either train or test.")
parser.add_argument(
    "--output-dir", type=str, default="./outputs/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--pred-output-file", type=str, default="results/result.txt",
    help="Save predicted results")
parser.add_argument(
    "--load-checkpoint", type=str, default='./outputs/',
    help="If specified, will load the pre-trained checkpoint from output_dir.")
parser.add_argument(
    "--epoch-id", type=str, default="-1",
    help="Epoch number")

args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

make_deterministic(config_model.random_seed)
# device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cuda:0,1")


def get_lr_multiplier(step: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    square-root decay.
    """
    multiplier = (min(1.0, step / warmup_steps) *
                  (1 / math.sqrt(max(step, warmup_steps))))
    return multiplier


class ModelWrapper(nn.Module):
    def __init__(self, model: Transformer, beam_width: int):
        super().__init__()
        self.model = model
        self.beam_width = beam_width

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:

        # print("batch.src_text_ids:", batch.src_text_ids, "batch.tgt_text_ids:", batch.tgt_text_ids, " -1: ", batch.tgt_text_ids[:,:-1].contiguous())
        # src_text_ids = batch.src_text_ids.to(device)
        # tgt_text_ids = batch.tgt_text_ids[:,:-1].contiguous().to(device)
        # labels = batch.tgt_text_ids[:,1:].contiguous().to(device)
        src_text_ids = batch['src_text_ids']
        tgt_text_ids = batch['tgt_text_ids'][:, :-1].contiguous()
        labels = batch['tgt_text_ids'][:, 1:].contiguous()
        sys.stdout.flush()
        # loss = self.model(encoder_input=batch.src_text_ids,
        #                  decoder_input=batch.tgt_text_ids[:,:-1].contiguous(),
        #                  labels=batch.tgt_text_ids[:,1:].contiguous())
        # print("src_text_ids:", src_text_ids)
        # print("tgt_text_ids:", tgt_text_ids)
        # print("labels:", labels)
        # print("src_text_ids.shape:", src_text_ids.shape)
        # print("tgt_text_ids.shape:", tgt_text_ids.shape)
        # print("labels.shape:", labels.shape)
        encoder_input_length = (src_text_ids != 0).int().sum(dim=1)
        loss = self.model(encoder_input=src_text_ids,
                          decoder_input=tgt_text_ids,
                          labels=labels,
                          encoder_input_length_max=encoder_input_length.max())
        return {"loss": loss}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.src_text_ids,
                                 beam_width=self.beam_width)
        if self.beam_width == 1:
            decoded_ids = predictions[0].sample_id
        else:
            decoded_ids = predictions["sample_id"][:, :, 0]
        return {"preds": decoded_ids}


def rm_begin_str_in_keys(str, dict):
    from collections import OrderedDict
    d = {}
    for k, v in dict.items():
        lenstr = len(str)
        if str == k[:lenstr]:
            k = k[lenstr:]
        d[k] = v
    return OrderedDict(d)


def main() -> None:
    """Entry point.
    """
    print("Start!!!")
    sys.stdout.flush()
    if args.run_mode == "train":
        train_data = tx.data.MultiAlignedData(config_data.train_data_params, device=device)
        # train_data = tx.data.MultiAlignedData(config_data.train_data_params, device=device)
        print("will data_iterator")
        data_iterator = tx.data.DataIterator({"train": train_data})
        print("data_iterator done")

        # Create model and optimizer
        model = Transformer(config_model, config_data, train_data.vocab('src'))
        model.to(device)
        print("device:", device)
        # print("vocab src1:", train_data.vocab('src').id_to_token_map_py)
        # print("vocab src2:", train_data.vocab('src').token_to_id_map_py)

        model = ModelWrapper(model, config_model.beam_width)
        model.load_state_dict(torch.load('./outputs/400w_transformer_base/checkpoint2.pt'))
        for name, parameters in model.named_parameters():
            print(name, ':', parameters)
            print('--------------------------------------------')
        if torch.cuda.device_count() > 1:
            # model = nn.DataParallel(model.cuda(), device_ids=[0, 1]).to(device)
            # model = MyDataParallel(model.cuda(), device_ids=[0, 1]).to(device)
            model = MyDataParallel(model.cuda()).to(device)

        lr_config = config_model.lr_config
        if lr_config["learning_rate_schedule"] == "static":
            init_lr = lr_config["static_lr"]
            scheduler_lambda = lambda x: 1.0
        else:
            init_lr = lr_config["lr_constant"]
            scheduler_lambda = functools.partial(
                get_lr_multiplier, warmup_steps=lr_config["warmup_steps"])
        optim = torch.optim.Adam(
            model.parameters(), lr=init_lr, betas=(0.9, 0.997), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)

        # resume
        # model.load_state_dict(torch.load(args.output_dir+f'/checkpoint{19}.pt'))
        # optim.load_state_dict(torch.load(args.output_dir + f'/optimizer{19}.pt'))
        # scheduler.load_state_dict(torch.load(args.output_dir + f'/scheduler{19}.pt'))

        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        def _save_epoch(epoch):

            checkpoint_name = f"checkpoint{epoch}.pt"
            print(f"saveing model...{checkpoint_name}")
            torch.save(model.state_dict(), output_dir / checkpoint_name)

            checkpoint_name = f"optimizer{epoch}.pt"
            print(f"saveing ... {checkpoint_name}")
            torch.save(optim.state_dict(), output_dir / checkpoint_name)

            checkpoint_name = f"scheduler{epoch}.pt"
            print(f"saveing ... {checkpoint_name}")
            torch.save(scheduler.state_dict(), output_dir / checkpoint_name)



        def _train_epoch(epoch):
            data_iterator.switch_to_dataset('train')
            model.train()
            # model.module.train()
            # print("after model.module.train")
            sys.stdout.flush()
            step = 0
            num_steps = len(data_iterator)
            loss_stats = []
            for batch in data_iterator:
                # print("batch:", batch)
                # batch = batch.to(device)
                return_dict = model(batch)
                # return_dict = model.module.forward(batch)
                loss = return_dict['loss']
                # print("loss:", loss)
                loss = loss.mean()
                # print("loss:", loss)
                # print("loss.item():", loss.item())
                loss_stats.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                config_data.display = 16
                if step % config_data.display == 0:
                    avr_loss = sum(loss_stats) / len(loss_stats)
                    ppl = utils.get_perplexity(avr_loss)
                    print(
                        f"epoch={epoch}, step={step}/{num_steps}, loss={avr_loss:.4f}, ppl={ppl:.4f}, lr={scheduler.get_lr()[0]}")
                    sys.stdout.flush()
                step += 1

        print("will train")
        for i in range(config_data.num_epochs):
            print("epoch i:", i)
            sys.stdout.flush()
            _train_epoch(i)
            _save_epoch(i)


    elif args.run_mode == "test":
        test_data = tx.data.MultiAlignedData(config_data.test_data_params, device=device)
        data_iterator = tx.data.DataIterator({"test": test_data})
        # print("test_data vocab src1 before load:", test_data.vocab('src').id_to_token_map_py)

        # Create model and optimizer
        model = Transformer(config_model, config_data, test_data.vocab('src'))

        model = ModelWrapper(model, config_model.beam_width)
        # print("state_dict:", model.state_dict())
        # model_loaded = torch.load(args.load_checkpoint)
        # print("model_loaded state_dict:", model_loaded)
        # model_loaded = rm_begin_str_in_keys("module.", model_loaded)
        # print("model_loaded2 state_dict:", model_loaded)
        model_loaded=torch.load(args.output_dir + f'/checkpoint{args.epoch_id}.pt')
        model.load_state_dict(model_loaded)
        # model.load_state_dict(torch.load(args.load_checkpoint))
        model.to(device)

        data_iterator.switch_to_dataset('test')
        model.eval()
        print("will predict !!!")
        sys.stdout.flush()

        fo = open(args.pred_output_file, "w")
        # print("test_data vocab src1:", test_data.vocab('src').id_to_token_map_py)
        # print("test_data vocab src2:", test_data.vocab('src').token_to_id_map_py)
        with torch.no_grad():
            for batch in data_iterator:
                # print("batch:", batch)
                return_dict = model.predict(batch)
                preds = return_dict['preds'].cpu()
                # print("preds:", preds)
                pred_words = tx.data.map_ids_to_strs(preds, test_data.vocab('src'))
                # src_words = tx.data.map_ids_to_strs(batch['src_text'], test_data.vocab('src'))
                src_words = [" ".join(sw) for sw in batch['src_text']]
                for swords, words in zip(src_words, pred_words):
                    print(str(swords) + "\t" + str(words))
                    fo.write(str(words) + "\n")
                # print(" ".join(batch.src_text) + "\t" + pred_words)
                # print(batch.src_text, pred_words)
                # fo.write(str(pred_words) + "\n")
                fo.flush()
        fo.close()


    else:
        raise ValueError(f"Unknown mode: {args.run_mode}")


if __name__ == "__main__":
    main()