# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import softmax
import texar.torch as tx
# from torch.nn.parallel.scatter_gather import scatter
from scatter_gather import scatter
from transformer_decoders import MyTransformerDecoder


class Transformer(nn.Module):
    r"""A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    def __init__(self, model_config, data_config, vocab: tx.data.Vocab, sentiment_tensor, device):
        super().__init__()

        self.config_model = model_config
        self.config_data = data_config
        self.vocab = vocab
        self.vocab_size = vocab.size

        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=self.vocab_size,
            hparams=self.config_model.emb)
        self.pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.config_data.max_decoding_length,
            hparams=self.config_model.position_embedder_hparams)

        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)
        self.decoder = MyTransformerDecoder(
            token_pos_embedder=self._embedding_fn,
            vocab_size=self.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=self.config_model.decoder)

        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=self.config_model.loss_label_confidence,
            tgt_vocab_size=self.vocab_size,
            ignore_index=0)
        self.device=device
        self.sentiment_tensor=sentiment_tensor.to(self.device)



    def _embedding_fn(self, tokens: torch.LongTensor,
                      positions: torch.LongTensor) -> torch.Tensor:
        token_device = tokens.get_device()
        # print("before self.word_embedder.embedding in _embedding_fn:", self.word_embedder.embedding)
        if token_device != self.word_embedder.embedding.get_device():
            print("still diff")
            # self.word_embedder._embedding = nn.Parameter(self.word_embedder.embedding.to(token_device), requires_grad=True)
            # self.word_embedder._embedding = scatter(self.word_embedder._embedding, [0, 1])
            print("after self.word_embedder.embedding in _embedding_fn:", self.word_embedder.embedding)
        word_embed = self.word_embedder(tokens)
        scale = self.config_model.hidden_dim ** 0.5
        pos_embed = self.pos_embedder(positions)
        return word_embed * scale + pos_embed


    def forward(self,  # type: ignore
                encoder_input: torch.Tensor,
                condition_generator:torch.nn.Module,
                posterior_mean_estimator:torch.nn.Module,
                posterior_variance_estimator:torch.nn.Module,
                key_value_combiner:torch.nn.Module,
                K: torch.Tensor,
                V: torch.Tensor,
                sentiment_classifier:Optional = None,
                tokenizer: Optional = None,
                merge_input: Optional[torch.Tensor] = None,
                decoder_input: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                beam_width: Optional[int] = None,
                encoder_input_length_max: Optional[int] = None):
        r"""Compute the maximum likelihood loss or perform decoding, depending
        on arguments.
        Args:
            encoder_input: the source sentence embedding, with the shape of
                `[batch_size, source_seq_length, input_dim]`.
            decoder_input: the target sentence embedding, with the shape of
                `[batch_size, target_seq_length, input_dim]`.
            labels: the target sentence labels, with the shape of
                `[batch_size, target_seq_length]`.
            beam_width: Used in beam search.
        :returns:
            - If both :attr:`decoder_input` and :attr:`labels` are both
              provided, the function enters training logic and returns the
              maximum likelihood loss.
            - Otherwise the function enters inference logic and returns the
              decoded sequence.
            - If `beam_width` > 1, beam search decoding is performed. Please
              refer to :meth:`texar.modules.TransformerDecoder.forward` for
              details on return types.
        """

        global value, KL_loss, key, encoder_representation
        batch_size = encoder_input.size(0)
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)
        if encoder_input_length_max is None:
            src_positions = torch.arange(
                encoder_input_length.max(), dtype=torch.long,
                device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)
        else:
            src_positions = torch.arange(
                encoder_input.size(1), dtype=torch.long,
                device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)

        # Source word embedding
        src_input_embedding = self._embedding_fn(encoder_input, src_positions)

        if merge_input is not None:
            new_input = merge_input
            new_input = self.word_embedder(new_input)

            c=condition_generator(new_input,self.word_embedder.embedding.get_device()) # torch.Size([1, bs, hiddem_dim])
            encoder_representation=c

            if decoder_input is not None:
                value=condition_generator(self.word_embedder(decoder_input),self.word_embedder.embedding.get_device())
                key=c.squeeze(dim=0).detach().cpu()
                value=value.squeeze(dim=0).detach()


            index = torch.matmul(c, K.detach().t())
            # print(f'index.shape:{index.shape}')    torch.Size([1, 96, 1000])
            index=index.argmax(dim=2).squeeze(dim=0)
            # print(f'index.shape:{index.shape}')    torch.Size([96])
            value_with_memory = torch.index_select(V, 0, index).unsqueeze(dim=0)
            # print(f'value_with_memory.shape:{value_with_memory.shape}')    torch.Size([1, 96, 512])

            prior_mean = value_with_memory
            posterior_mean=posterior_mean_estimator(c)
            posterior_log_variance_square=posterior_variance_estimator(c)
            sampled_c=posterior_mean+posterior_log_variance_square*torch.randn_like(posterior_log_variance_square) # reparameterization
            # c=key_value_combiner(c,sampled_c)
            c=sampled_c
            KL_loss=0.5*((posterior_mean-prior_mean)*(posterior_mean-prior_mean)+torch.exp(posterior_log_variance_square)-posterior_log_variance_square-1).mean()  # https://spaces.ac.cn/archives/5253#%E6%9D%A1%E4%BB%B6VAE
            # print(f'c.shape:{c.shape}')    torch.Size([1, 96, 512])


            first_token=torch.unsqueeze(src_input_embedding[:,0,:],dim=1)
            rest_tokens=src_input_embedding[:,1:,:]
            src_with_emotional=torch.cat((first_token,torch.transpose(c, 0, 1),rest_tokens),dim=1)

            encoder_input_length+=1
            encoder_output = self.encoder(
                inputs=src_with_emotional, sequence_length=encoder_input_length)



        if decoder_input is not None and labels is not None: # train-mode (teacher-forcing training)
            outputs = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                inputs=decoder_input,
                decoding_strategy="train_greedy",
                my_emb_fn=self._embedding_fn) # outputs.logits.shape: torch.Size([batch_size, max_target_seq_length, vocab_size+2])
            label_lengths = (labels != 0).long().sum(dim=1) # torch.Size([batch_size])
            is_target = (labels != 0).float() # torch.Size([batch_size, max_target_seq_length])
            mle_loss = self.smoothed_loss_func(
                outputs.logits, labels, label_lengths) # torch.Size([batch_size, max_target_seq_length])
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()

            vocab = tx.data.Vocab('data/data_v15_d1g10_transductive_for_base_ori/dict.txt',eos_token='[SEP]')
            # print(f'outputs.logits.shape:{outputs.logits.shape}')
            pred_batch_sentence=[]
            for i in range(outputs.logits.shape[0]):
                pred_sentence=str(tx.data.map_ids_to_strs(outputs.logits.argmax(dim=2)[i].cpu(), vocab, strip_eos=None)) # str(<class 'numpy.str_'>) = <class 'str'>
                pred_batch_sentence.append(pred_sentence)
            res = tokenizer.batch_encode_plus(pred_batch_sentence, pad_to_max_length=True)
            input_ids = torch.tensor(res['input_ids'], dtype=torch.long, device=self.device)
            # print(f'input_ids.shape:{input_ids.shape}')
            token_type_ids = res['token_type_ids']
            attention_mask = res['attention_mask']
            Bert_outputs = sentiment_classifier(input_ids=input_ids, attention_mask=torch.tensor(attention_mask, device=self.device),
                                 token_type_ids=torch.tensor(token_type_ids, device=self.device))
            sentiment_score = softmax(Bert_outputs[0], dim=1)[:, 1]-0.5 # shape: torch.Size([batch_size])

            # R_t=self.linearRegression(outputs.logits.max(dim=2)[0])
            # representation_from_bert_last_layer=Bert_outputs[1][-2].detach()
            # print(f'representation_from_bert_last_layer:{representation_from_bert_last_layer}')
            # print(f'outputs.logits.argmax(dim=2):{outputs.logits.argmax(dim=2)}')
            a,b=outputs.logits.argmax(dim=2).shape
            token_level_sentiment_score = torch.index_select(self.sentiment_tensor,0,outputs.logits.argmax(dim=2).view(-1,1).squeeze()).view(a,b)
            # print(f'token_level_sentiment_score:{token_level_sentiment_score}')
            # P_wt=torch.gather(softmax(outputs.logits, dim=2), 2, labels.unsqueeze(2)).squeeze(2)
            P_wt=softmax(outputs.logits, dim=2).max(dim=2)[0]


            def _RL_loss(R, P_wt):
                '''
                R: sentiment_score for batch_size sentences  (shape:[batch_size])
                Rt: sentiment_score for the t-th character in the sentence of a batch (shape:[batch_size,sentence_length])
                P_wt: the probability for the t-th character in the sentence of a batch (shape:[batch_size,sentence_length])
                reduce: return the loss in a scalar from or in a tensor form
                '''

                # loss = -((R-0.1562) * torch.log(P_wt.float()+1e-9)).mean(dim=1)
                loss = -(R*torch.log(P_wt.float()+1e-9)).mean(dim=1)
                return loss
            # print(f'token_level_sentiment_score.shape:{token_level_sentiment_score.shape[1]}')
            # print(f'P_wt.shape:{P_wt.shape[1]}')
            RL_loss=_RL_loss(token_level_sentiment_score.detach(),P_wt)

            return RL_loss.mean(),mle_loss,KL_loss,sentiment_score.detach(),key,value

        else: # test-mode
            start_tokens = encoder_input.new_full(
                (batch_size,), self.vocab.bos_token_id)

            top_k = 20 # ori:5
            temperature = 0.7
            decoding_helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.vocab.eos_token_id,
                top_k=top_k,
                softmax_temperature=temperature)

            predictions = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                beam_width=beam_width,
                length_penalty=self.config_model.length_penalty,
                start_tokens=start_tokens,
                end_token=self.vocab.eos_token_id,
                max_decoding_length=self.config_data.max_decoding_length,
                decoding_strategy="infer_greedy",
                helper=decoding_helper
            )
            # Uses the best sample by beam search
            return predictions



class condition_generator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.linear1 = torch.nn.Linear(self.hidden_dim, 6)  # 6: Affect-Driven Dialog Generation
        self.rnn = torch.nn.RNN(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.linear2 = torch.nn.Linear(6, self.hidden_dim)

    def forward(self, input: torch.Tensor,device):
        h_0 = torch.rand(1, input.size()[0], self.hidden_dim).to(device)
        _, c = self.rnn(input, h_0)
        c = self.linear1(c)
        return self.linear2(c)

class combiner(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.linear = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def forward(self, condition, value):
        return self.linear(torch.cat([condition,value],dim=2))



# class SentimentClassifier(nn.Module):
#   def __init__(self):
#     super(SentimentClassifier, self).__init__()
#
#   def forward(self, input_ids:torch.Tensor):
#     batch_size=input_ids.size()[0]
#     return torch.rand(batch_size,1).cuda()



class TokenLevelSentimentAnalysis(nn.Module):
    def __init__(self):
        super(TokenLevelSentimentAnalysis, self).__init__()
        self.linear1 = nn.Linear(768, 5)
        self.linear2 = nn.Linear(5, 2)
        self.ReLU=nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, Bert_representation):
        return softmax(self.linear2(self.ReLU(self.linear1(Bert_representation))),dim=2)

class estimator(nn.Module):
    def __init__(self):
        super(estimator, self).__init__()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)
        self.ReLU=nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear2(self.ReLU(self.linear1(input)))


class LabelSmoothingLoss(nn.Module):
    r"""With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: torch.Tensor

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self,  # type: ignore
                output: torch.Tensor,
                target: torch.Tensor,
                label_lengths: torch.LongTensor) -> torch.Tensor:
        r"""Compute the label smoothing loss.
        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = (output.size(), target.size())
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])

        return tx.losses.sequence_softmax_cross_entropy(
            labels=model_prob,
            logits=output,
            sequence_length=label_lengths,
            average_across_batch=False,
            sum_over_timesteps=False,
        )