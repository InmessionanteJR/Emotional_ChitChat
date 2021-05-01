# from transformers import pipeline
# classifier = pipeline('sentiment-analysis')
# res=classifier(['你们速度快的我都来不及看电视'])
# res=[{'label': 'NEGATIVE', 'score': 0.8849299550056458}, {'label': 'NEGATIVE', 'score': 0.8642347455024719}]
# print(res)

import texar.torch as tx
from transformers import BertForSequenceClassification,BertTokenizerFast,AdamW,get_linear_schedule_with_warmup
import torch
from torch.nn.functional import softmax
import sys
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-mode", type=str, default="train",
    help="Either train or test or generate.")
parser.add_argument(
    "--output-dir", type=str, default=os.getcwd()+"/Bert/checkpoints/",
    help="Path to save the trained model, losses (in train mode) and predicted labels (in generate mode).")
parser.add_argument(
    "--pretrained-dir", type=str, default=os.getcwd()+"/Bert/checkpoints/",
    help="Path to load the pretrained model.")
parser.add_argument(
    "--output-name", type=str, default="default_output_name",
    help="To name sentiment score files with marks.")
parser.add_argument(
    "--texar-src-name", type=str, default="default_texar_src_name",
    help="Test dataset's (the file which we want to know sentiment scores) name.")
parser.add_argument(
    "--Bert-mark", type=str, default="default_Bert_mark",
    help="Help to name pred_labels' paths, see .sh file for details.")
args = parser.parse_args()

epoch_num=20
lr = 1e-6
max_grad_norm = 1.0
num_training_steps = 100000
num_warmup_steps = 1600
# warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

max_sentence_length=128 # Bert default:512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparams_train={
    'datasets': [
        {'files': 'corpus_train.txt', 'vocab_file': 'dict.txt', 'data_name': 'train_corpus'},
        # {'files': 'corpus_test.txt', 'vocab_file': 'dict.txt', 'data_name': 'test_corpus'},
        {'files': 'label_train.txt', 'vocab_file': 'dict.txt', 'data_name': 'train_label'}
        # {'files': 'label_test.txt', 'vocab_file': 'dict.txt', 'data_name': 'test_label'}
    ],
    'batch_size': 96,
    'shuffle': False
}
hparams_test_weibo_senti={
    'datasets': [
        {'files': 'corpus_test.txt', 'vocab_file': 'dict.txt', 'data_name': 'test_corpus'},
        {'files': 'label_test.txt', 'vocab_file': 'dict.txt', 'data_name': 'test_label'}
    ],
    'batch_size': 32,
    'shuffle': False
}
# hparams_test_ChnSentiCorp={
#     'datasets': [
#         {'files': 'ChnSentiCorp_corpus_test.txt', 'vocab_file': 'dict.txt', 'data_name': 'test_corpus'},
#         {'files': 'ChnSentiCorp_label_test.txt', 'vocab_file': 'dict.txt', 'data_name': 'test_label'}
#     ],
#     'batch_size': 16,
#     'shuffle': False
# }
hparams_generate={
    'datasets': [
        # {'files': '../data/train400w.src', 'vocab_file': 'dict.txt', 'data_name': 'data'}
        {'files': os.getcwd()+'/results/'+args.texar_src_name, 'vocab_file': os.getcwd()+'/Bert/dict.txt', 'data_name': 'data'}
    ],
    'batch_size': 64,
    'shuffle': False
}

# bert_model = BertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir='.',num_labels=2)
bert_model = BertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir=os.getcwd()+'/Bert/pretrained',num_labels=2)
bert_model.to(device)

# for name, value in bert_model.named_parameters():
# for name, value in bert_model.named_parameters():
#     print('name: {0},\t shape: {1}'.format(name, value.shape))

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', cache_dir='./Bert/pretrained')

optimizer = AdamW(bert_model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

output_dir = Path(os.getcwd()+'/'+args.output_dir)
if not output_dir.exists():
    output_dir.mkdir()

if args.run_mode == "train":
    data_train = tx.data.MultiAlignedData(hparams_train)
    iterator_train = tx.data.DataIterator(data_train)

    Loss=[]
    for epoch in range(epoch_num):
        step=0
        num_steps = len(iterator_train)
        for batch in iterator_train:
            bert_model.train()
            batch_sentence=[]
            batch_label=[]
            for i in range(len(batch['train_corpus_text'])):
                sentence=''
                for j in range(1,batch['train_corpus_length'][i].item()-1):
                    sentence+=batch['train_corpus_text'][i][j]
                batch_sentence.append(sentence)
            for j in range(len(batch['train_label_text'])):
                batch_label.append(int(batch['train_label_text'][j][1]))
            labels=torch.tensor(batch_label,dtype=torch.long,device=device)

            res = tokenizer.batch_encode_plus(batch_sentence, pad_to_max_length=True)
            input_ids = torch.tensor(res['input_ids'], dtype=torch.long,device=device)
            token_type_ids = res['token_type_ids']
            attention_mask = res['attention_mask']
            outputs = bert_model(input_ids=input_ids, attention_mask=torch.tensor(attention_mask,device=device), token_type_ids=torch.tensor(token_type_ids,device=device),
                                 labels=labels.view(-1,1))
            loss, logits = outputs[:2]
            acc=(labels==logits.argmax(dim=1)).sum().item()/len(labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            display=16
            if step % display == 0:
                avr_loss = loss.mean().item()
                Loss.append(avr_loss)
                print(
                    f"epoch={epoch}, step={step}/{num_steps}, loss={avr_loss:.4f}, acc={acc*100}%, lr={scheduler.get_lr()[0]}")
                sys.stdout.flush()
            step += 1

        checkpoint_name = f"checkpoint{epoch}.pt"
        print(f"saveing User...{checkpoint_name}")
        torch.save(bert_model.state_dict(),  output_dir / checkpoint_name)

    print("start saving loss!")
    torch.save(Loss, output_dir / 'loss')
    print("successfully save loss!")


elif args.run_mode == "test":
    data_test = tx.data.MultiAlignedData(hparams_test_weibo_senti)
    iterator_test = tx.data.DataIterator(data_test)

    for epoch in range(epoch_num): # test each checkpoint's result
        step=0
        num_steps = len(iterator_test)
        checkpoint_name = f"checkpoint{epoch}.pt"
        bert_model.load_state_dict(torch.load(output_dir / checkpoint_name))
        ACC=[]
        for batch in iterator_test:
            bert_model.eval()
            batch_sentence = []
            batch_label = []
            for i in range(len(batch['test_corpus_text'])):
                sentence = ''
                for j in range(1, batch['test_corpus_length'][i].item() - 1):
                    sentence += batch['test_corpus_text'][i][j]
                batch_sentence.append(sentence)
            for j in range(len(batch['test_label_text'])):
                batch_label.append(int(batch['test_label_text'][j][1]))
            labels = torch.tensor(batch_label, dtype=torch.long, device=device)

            res = tokenizer.batch_encode_plus(batch_sentence, pad_to_max_length=True)
            input_ids = torch.tensor(res['input_ids'], dtype=torch.long, device=device)
            token_type_ids = res['token_type_ids']
            attention_mask = res['attention_mask']
            outputs = bert_model(input_ids=input_ids, attention_mask=torch.tensor(attention_mask, device=device),
                                 token_type_ids=torch.tensor(token_type_ids, device=device),
                                 labels=labels)
            loss, logits = outputs[:2]
            acc = (labels == logits.argmax(dim=1)).sum().item() / len(labels)
            ACC.append(acc)

            display = 16
            if step % display == 0:
                avr_loss = loss.mean().item()
                print(
                    f"epoch={epoch}, step={step}/{num_steps}, loss={avr_loss:.4f}, acc={acc * 100}%")
                sys.stdout.flush()
            step += 1

        average_acc=sum(ACC)/len(ACC)
        print(f"----------------------------------The average accuracy of testset in epoch {epoch} is {average_acc*100}%!----------------------------------")


elif args.run_mode == "generate":
    data_generate = tx.data.MultiAlignedData(hparams_generate)
    iterator_test = tx.data.DataIterator(data_generate)

    # for epoch in range(epoch_num): # generate lables using multiple checkpoints to compare results
    epoch=4 # epoch 4 in weibo_senti_100k is the best epoch
    sentiment_score_list=[]
    step=0
    num_steps = len(iterator_test)
    checkpoint_name = f"checkpoint{epoch}.pt"
    bert_model.load_state_dict(torch.load(output_dir / checkpoint_name))
    for batch in iterator_test:
        bert_model.eval()
        batch_sentence = []
        for i in range(len(batch['data_text'])):
            sentence = ''
            for j in range(1, batch['data_length'][i].item() - 1):
                sentence += batch['data_text'][i][j]
            batch_sentence.append(sentence)
        res = tokenizer.batch_encode_plus(batch_sentence, pad_to_max_length=True)
        input_ids = torch.tensor(res['input_ids'], dtype=torch.long, device=device)
        token_type_ids = res['token_type_ids']
        attention_mask = res['attention_mask']
        outputs = bert_model(input_ids=input_ids, attention_mask=torch.tensor(attention_mask, device=device),
                             token_type_ids=torch.tensor(token_type_ids, device=device))
        logits = outputs[0]
        sentiment_score=softmax(logits, dim=1)[:,1]
        # print(sentiment_score)
        for i in range(sentiment_score.shape[0]):
            sentiment_score_list.append(sentiment_score[i].item())
        display = 32
        if step % display == 0:
            print(
                f"Using checkpoint{epoch} to generate, step={step}/{num_steps}")
            sys.stdout.flush()
        step += 1

    print("start saving pred_labels!")
    # name = f"pred_labels_{epoch}_1022"
    name = f"pred_labels_{args.output_name}"

    save_dir = Path(output_dir/args.Bert_mark)
    if not save_dir.exists():
        save_dir.mkdir()
    torch.save(sentiment_score_list, output_dir/args.Bert_mark/name)
    print("successfully save pred_labels!")