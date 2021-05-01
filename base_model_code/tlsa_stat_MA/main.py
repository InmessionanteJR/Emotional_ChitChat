import argparse
import functools
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import math
import time


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import texar.torch as tx
from texar.torch.run import make_deterministic # Make experiment deterministic by using specific random seeds
from transformers import BertForSequenceClassification,BertTokenizerFast
from data_parallel import MyDataParallel

from model import Transformer, condition_generator, combiner
from sklearn.cluster import KMeans
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
    help="Path to save the trained model and losses.")
parser.add_argument(
    "--pred-output-file", type=str, default="results/result.txt",
    help="Path to save predicted results")
parser.add_argument(
    "--epoch-id", type=str, default="-1",
    help="Epoch number")

args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

make_deterministic(config_model.random_seed) # set seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(args.output_dir+'/tensorboard_log/')

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
                batch: tx.data.Batch,condition_generator,key_value_combiner,K,V,sentiment_classifier,tokenizer) -> Dict[str, torch.Tensor]:
                # batch: tx.data.Batch,condition_generator) -> Dict[str, torch.Tensor]:

        src_text_ids = batch['src_text_ids']
        tgt_text_ids = batch['tgt_text_ids'][:,:-1].contiguous() # 单独开辟一块新的内存，具体可见 https://zhuanlan.zhihu.com/p/64551412
                                    # actually -1 makes nonsense due to the configuration of Texar
        merge_text_ids=batch['merge_text_ids']
        labels = batch['tgt_text_ids'][:,1:].contiguous()
        sys.stdout.flush()

        encoder_input_length = (src_text_ids != 0).int().sum(dim=1) # encoder_input_length=[该batch中第一句的长度，第二句的长度，... ，最后一句的长度]
        loss = self.model(encoder_input=src_text_ids,
                          condition_generator=condition_generator,
                          key_value_combiner=key_value_combiner,
                          K=K, V=V,
                          sentiment_classifier=sentiment_classifier,
                          tokenizer=tokenizer,
                          merge_input=merge_text_ids,
                          decoder_input=tgt_text_ids,
                          labels=labels,
                          encoder_input_length_max=encoder_input_length.max())
        return {"loss": loss}


    def predict(self, batch: tx.data.Batch,condition_generator,key_value_combiner,K,V) -> Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.src_text_ids,
                                 condition_generator=condition_generator,
                                 key_value_combiner=key_value_combiner,
                                 K=K, V=V,
                                 merge_input=batch.merge_text_ids,
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
        print("Will start data iterator!")
        data_iterator = tx.data.DataIterator({"train": train_data})
        print("Data_iterator done!\n")


        # Create user and optimizer
        # SentimentAnalysis=TokenLevelSentimentAnalysis().to(device)

        # Bert_based_sentiment_classifier
        sentiment_classifier=BertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir='./Bert/pretrained',num_labels=2, output_hidden_states=True).to(device)
        sentiment_classifier.load_state_dict(torch.load('./Bert/checkpoints/weibo_senti_100k/checkpoint4.pt'))

        # System_conditional_generator
        System_condition_generator=condition_generator(config_model.hidden_dim).to(device)
        System_condition_generator.load_state_dict(torch.load('./outputs/System_condition_generator_checkpoint.pt'))
        # System_condition_generator.load_state_dict(torch.load(args.output_dir+f'/System_condition_generator_checkpoint{2}.pt'))

        # key(i.e, condition)-value combiner
        key_value_combiner=combiner(config_model.hidden_dim).to(device)
        # key_value_combiner.load_state_dict(torch.load(args.output_dir + f'/Combiner_checkpoint{2}.pt'))


        # System
        sentiment_tensor=torch.load("data/data_v15_d1g10_transductive_for_base_ori/sentiment_tensor")
        System = Transformer(config_model, config_data, train_data.vocab('src'),sentiment_tensor,device).to(device)
        System = ModelWrapper(System, config_model.beam_width)
        # System.load_state_dict(torch.load(args.output_dir+f'/System_checkpoint{2}.pt'))
        System.load_state_dict(torch.load('./outputs/System_checkpoint.pt'))

        #BertTokenizer
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', cache_dir='./Bert/pretrained')

        # kmeans_model
        kmean_model = KMeans(n_clusters=1000, max_iter=30,n_init=10, n_jobs=-1)

        K=torch.load('outputs/extract_centers/cluster_key_centers').float().to(device) # cluster centers for keys
        V=torch.load('outputs/extract_centers/cluster_value_centers').float().to(device) # weighted representations for each key
        # K = torch.load(args.output_dir+f'/cluster_key_centers_{2}').float().to(device)  # cluster centers for keys
        # V = torch.load(args.output_dir+f'/cluster_value_centers_{2}').float().to(device)  # weighted representations for each key


        if torch.cuda.device_count() > 1:
            System = MyDataParallel(System.cuda()).to(device)
            print('WARNING: Using mult-gpus may induce a worse result!!!')

        lr_config = config_model.lr_config
        if lr_config["learning_rate_schedule"] == "static":
            init_lr = lr_config["static_lr"]
            scheduler_lambda = lambda x: 1.0
        else:
            init_lr = lr_config["lr_constant"]
            scheduler_lambda = functools.partial(
                get_lr_multiplier, warmup_steps=lr_config["warmup_steps"])

        optim = torch.optim.Adam([
                {'params': key_value_combiner.parameters()},
                {'params': System.parameters()},
                {'params': System_condition_generator.parameters()}
            ], lr=init_lr, betas=(0.9, 0.997), eps=1e-9)
        # optim = torch.optim.Adam([
        #         {'params': System.parameters()},
        #         {'params': System_condition_generator.parameters()}
        #     ], lr=init_lr, betas=(0.9, 0.997), eps=1e-9)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)
        # scheduler_2 = torch.optim.lr_scheduler.LambdaLR(optim_Sentiment_Analysis, scheduler_lambda)

        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        def _save_epoch(epoch):
            checkpoint_name = f"System_condition_generator_checkpoint{epoch}.pt"
            print(f"saveing ... {checkpoint_name}")
            torch.save(System_condition_generator.state_dict(), output_dir / checkpoint_name)
            # if i==config_data.num_epochs-1:
            #     checkpoint_name = f"System_checkpoint{epoch}.pt"
            #     print(f"saveing ... {checkpoint_name}")
            #     torch.save(System.state_dict(), output_dir / checkpoint_name)

            checkpoint_name = f"System_checkpoint{epoch}.pt"
            print(f"saving ... {checkpoint_name}")
            torch.save(System.state_dict(), output_dir / checkpoint_name)

            checkpoint_name = f"Combiner_checkpoint{epoch}.pt"
            print(f"saving ... {checkpoint_name}")
            torch.save(key_value_combiner.state_dict(), output_dir / checkpoint_name)

            # checkpoint_name = f"SentimentAnalysis_checkpoint{epoch}.pt"
            # print(f"saving ... {checkpoint_name}")
            # torch.save(SentimentAnalysis.state_dict(), output_dir / checkpoint_name)

        def _save_optimizer(epoch):
            checkpoint_name = f"optimizer{epoch}.pt"
            print(f"saveing ... {checkpoint_name}")
            torch.save(optim.state_dict(), output_dir / checkpoint_name)

            checkpoint_name = f"scheduler{epoch}.pt"
            print(f"saveing ... {checkpoint_name}")
            torch.save(scheduler.state_dict(), output_dir / checkpoint_name)



        def _save_loss():
            print("Start saving loss!")
            name1="RL_loss"
            name2="mle_loss"
            name3="mixed_loss"
            name4="avg_sentiment_scores"
            torch.save(RL_LOSS, output_dir / name1)
            torch.save(MLE_LOSS, output_dir / name2)
            torch.save(MIXED_LOSS, output_dir / name3)
            torch.save(SENTIMENT_SCORES, output_dir / name4)
            print("Successfully saved loss!")

        def _save_ppl():
            print("Start saving ppl!")
            name="ppl"
            torch.save(PPL, output_dir / name)
            print("Successfully saved ppl!")

        def _caculate_kmean_centers(kmean_model,epoch):
            torch.save(System.model.key, output_dir / f"System.model.key_{epoch}")
            torch.save(System.model.value, output_dir / f"System.model.value_{epoch}")

            System.model.key=System.model.key.cpu()
            System.model.value=System.model.value.cpu()

            start = time.time()
            print(f'len(System.model.key):{System.model.key.shape[0]}')
            print(f'len(System.model.value):{System.model.value.shape[0]}')
            kmeans=kmean_model.fit(System.model.key)
            torch.save(kmean_model, output_dir / f'kmean_model_{epoch}')
            cluster_centers=torch.tensor(kmeans.cluster_centers_)
            value_in_clusters={}
            print(f'len(kmeans.labels_):{len(kmeans.labels_)}')

            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] in value_in_clusters:
                    value_in_clusters[kmeans.labels_[i]] = torch.cat(
                        (value_in_clusters[kmeans.labels_[i]], System.model.value[i].unsqueeze(dim=0)), dim=0)
                else:
                    value_in_clusters[kmeans.labels_[i]] = System.model.value[i].unsqueeze(dim=0)

            tmp = 0
            first = True
            value_in_clusters = sorted(value_in_clusters.items(), key=lambda d: d[0])

            for i, value in enumerate(value_in_clusters):
                value = value[1]
                # print(f'value.shape(0) for key:{i+1}:{value.shape[0]}')
                avg_value = value.mean(dim=0).unsqueeze(dim=0)
                # print(f'avg_value.shape:{avg_value.shape}')
                if first:
                    tmp = avg_value
                    first = False
                else:
                    tmp = torch.cat((tmp, avg_value), dim=0)

            end = time.time()
            print(f'kmean_time:{end-start}')

            torch.save(torch.tensor(kmeans.cluster_centers_), output_dir / f'cluster_key_centers_{epoch}')
            torch.save(tmp, output_dir / f'cluster_value_centers_{epoch}')

            System.model.key=None
            System.model.value=None
            return cluster_centers.float().to(device),tmp.float().to(device)

        def _train_epoch(epoch,K,V):
            data_iterator.switch_to_dataset('train')
            # Re-initializes the iterator of a given dataset and starts iterating
            # over the dataset (from the beginning).

            System.train()
            System_condition_generator.train()

            sys.stdout.flush()
            step = 0
            num_steps=len(data_iterator)
            print('num_steps:'+str(num_steps))

            mle_loss_stats = []
            RL_loss_stats = []
            mixed_loss_stats = []
            sentiment_scores_stats = []


            for batch in data_iterator:
                return_dict = System(batch,System_condition_generator,key_value_combiner,K,V,sentiment_classifier,tokenizer)

                RL_loss,mle_loss,sentiment_scores= return_dict['loss']
                mixed_loss=RL_loss*0.90+mle_loss*0.10
                # mixed_loss=RL_loss
                optim.zero_grad()
                mixed_loss.backward()
                nn.utils.clip_grad_norm_(System.parameters(), 5)
                nn.utils.clip_grad_norm_(System_condition_generator.parameters(), 5)
                optim.step()
                scheduler.step()

                mle_loss_stats.append(mle_loss.item())
                RL_loss_stats.append(RL_loss.item())
                mixed_loss_stats.append(mixed_loss.item())
                sentiment_scores_stats.append(sentiment_scores.mean())

                if step % config_data.display == 0:
                    avr_mle_loss = sum(mle_loss_stats) / len(mle_loss_stats)
                    avr_RL_loss = sum(RL_loss_stats) / len(RL_loss_stats)
                    avr_mixed_loss = sum(mixed_loss_stats) / len(mixed_loss_stats)
                    avg_sentiment_scores=sum(sentiment_scores_stats)/len(sentiment_scores_stats)
                    ppl = utils.get_perplexity(avr_mle_loss)

                    writer.add_scalars('loss', {'mle_loss': avr_mle_loss,
                                                'RL_loss': avr_RL_loss,
                                                'mixed_loss': avr_mixed_loss,
                                                'avg_sentiment_scores': avg_sentiment_scores,
                                                'ppl': ppl}, step+1+i*len(data_iterator))


                    MLE_LOSS.append(avr_mle_loss)
                    RL_LOSS.append(avr_RL_loss)
                    MIXED_LOSS.append(avr_mixed_loss)
                    PPL.append(ppl)
                    SENTIMENT_SCORES.append(avg_sentiment_scores)
                    print(f"epoch={epoch}, step={step}/{num_steps}, mle_loss={avr_mle_loss:.4f}, ppl={ppl:.4f}, RL_loss={avr_RL_loss:.4f}, mixed_loss={avr_mixed_loss:.4f}, avg_sentiment_scores:{avg_sentiment_scores},lr={scheduler.get_last_lr()[0]}")

                    sys.stdout.flush()
                    mle_loss_stats.clear()
                    RL_loss_stats.clear()
                    mixed_loss_stats.clear()
                    sentiment_scores_stats.clear()


                if step % 1000 == 0:
                    print('Empty cache!')
                    del return_dict,RL_loss,mle_loss,mixed_loss
                    torch.cuda.empty_cache()

                step += 1

        print("will train\n")
        MLE_LOSS=[]
        RL_LOSS=[]
        MIXED_LOSS=[]
        PPL=[]
        SENTIMENT_SCORES=[]

        for i in range(config_data.num_epochs):
            print("epoch :", i)
            sys.stdout.flush()
            train_start = time.time()
            _train_epoch(i,K,V)
            train_end = time.time()
            print(f'training time:{train_end - train_start}')
            _save_epoch(i)
            _save_loss()
            _save_ppl()
            _save_optimizer(i)
            K,V=_caculate_kmean_centers(kmean_model,i)
            print('\n')



    elif args.run_mode == "test":
        test_data = tx.data.MultiAlignedData(config_data.test_data_params, device=device)
        data_iterator = tx.data.DataIterator({"test": test_data})

        # Create model
        sentiment_tensor=torch.load("data/data_v15_d1g10_transductive_for_base_ori/sentiment_tensor")
        model = Transformer(config_model, config_data, test_data.vocab('src'),sentiment_tensor,device)
        model = ModelWrapper(model, config_model.beam_width)
        model_loaded = torch.load(args.output_dir+f'/System_checkpoint{args.epoch_id}.pt')
        model.load_state_dict(model_loaded)
        model.to(device)

        # Create condition generator
        System_condition_generator=condition_generator(config_model.hidden_dim).to(device)
        System_condition_generator.load_state_dict(torch.load(args.output_dir+f'/System_condition_generator_checkpoint{args.epoch_id}.pt'))

        # key(i.e, condition)-value combiner
        key_value_combiner = combiner(config_model.hidden_dim).to(device)
        key_value_combiner.load_state_dict(torch.load(args.output_dir+f'/Combiner_checkpoint{args.epoch_id}.pt'))

        K = torch.load(args.output_dir+f'/cluster_key_centers_{args.epoch_id}').float().to(device)  # cluster centers for keys
        V = torch.load(args.output_dir+f'/cluster_value_centers_{args.epoch_id}').float().to(device)  # weighted representations for each key

        data_iterator.switch_to_dataset('test')
        model.eval()
        print("will predict !!!")
        sys.stdout.flush()


        fo = open(args.pred_output_file, "w")
        with torch.no_grad():
            for batch in data_iterator:
                return_dict = model.predict(batch,System_condition_generator,key_value_combiner,K,V)
                preds = return_dict['preds'].cpu()
                print("preds:", preds)
                pred_words = tx.data.map_ids_to_strs(preds, test_data.vocab('src'))

                src_words = [" ".join(sw) for sw in batch['src_text']]
                for swords, words in zip(src_words, pred_words):
                    print(str(swords) + "\t" + str(words))
                    fo.write(str(words) + "\n")

                fo.flush()
        fo.close()


    else:
        raise ValueError(f"Unknown mode: {args.run_mode}")


if __name__ == "__main__":
    main()