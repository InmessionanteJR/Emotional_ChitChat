import os
max_decoding_length = 200
num_epochs=20
display=16
accumulation_steps=2


# vocab_length=12009
# vocab_length+=3 # BOS,EOS,OOV



data_root_dir="data/data_v15_d1g10_transductive_for_base_ori/"



train_src_file='weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_h125ktrain_shuffled_v21.train.ori.src'
train_tgt_file='weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_h125ktrain_shuffled_v21.train.ori.tgt'
train_merge_file='merge.txt'


train_src_vocab='dict.txt'
train_tgt_vocab=train_src_vocab

# hparams={
#     'datasets': [
#         {'files': 'a.txt', 'vocab_file': 'v.a', 'data_name': 'x'},
#         {'files': 'b.txt', 'vocab_file': 'v.b', 'data_name': 'y'},
#         {'files': 'c.txt', 'data_type': 'int', 'data_name': 'z'}
#     ]
#     'batch_size': 1
# }
# data = MultiAlignedData(hparams)
# iterator = DataIterator(data)
#
# for batch in iterator:
    # batch contains the following
    # batch == {
    #    'x_text': [['<BOS>', 'x', 'sequence', '<EOS>']],
    #    'x_text_ids': [['1', '5', '10', '2']],
    #    'x_length': [4]
    #    'y_text': [['<BOS>', 'y', 'sequence', '1', '<EOS>']],
    #    'y_text_ids': [['1', '6', '10', '20', '2']],
    #    'y_length': [5],
    #    'z': [1000],
    # }


train_data_params={
    'datasets':[
        {'files': os.path.join(data_root_dir, train_src_file),
            'vocab_file': os.path.join(data_root_dir, train_src_vocab),
            'data_name': 'src'},
        {'files': os.path.join(data_root_dir, train_tgt_file),
            'vocab_file': os.path.join(data_root_dir, train_tgt_vocab),
            'data_name': 'tgt'},
        {'files': os.path.join(data_root_dir, train_merge_file),
            'vocab_file': os.path.join(data_root_dir, train_tgt_vocab),
            'data_name': 'merge'},
        {'files': os.path.join(data_root_dir, 'src_label.txt'),
         'vocab_file': os.path.join(data_root_dir, train_tgt_vocab),
         'data_name': 'src_score'},
        {'files': os.path.join(data_root_dir, 'tgt_label.txt'),
         'vocab_file': os.path.join(data_root_dir, train_tgt_vocab),
         'data_name': 'tgt_score'}
    ],
    'batch_size': 32
    # 'shuffle': False # default=True
}


# train_data_params={
#     'datasets':[
#         {'files': 'data/train400w.src',
#             'vocab_file': 'data/dict400w',
#             'data_name': 'src'},
#         {'files': 'data/train400w.tgt',
#             'vocab_file': 'data/dict400w',
#             'data_name': 'tgt'},
#         {'files': 'data/train400w.tgt',
#             'vocab_file': 'train400w.src_tgt',
#             'data_name': 'merge'},
#         # {'files': os.path.join(data_root_dir, 'src_label.txt'),
#         #  'vocab_file': os.path.join(data_root_dir, train_tgt_vocab),
#         #  'data_name': 'src_score'},
#         # {'files': os.path.join(data_root_dir, 'tgt_label.txt'),
#         #  'vocab_file': os.path.join(data_root_dir, train_tgt_vocab),
#         #  'data_name': 'tgt_score'}
#     ],
#     'batch_size': 64
#     # 'shuffle': False # default=True
# }


# test
test_src_file='weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_h125ktrain_shuffled_v21_wo_train.test.ori.src'
# test_src_file='wyl_test.src'
test_merge_file='merge.txt'

test_data_params={
    'datasets':[
        {'files': os.path.join(data_root_dir, test_src_file), 
            'vocab_file': os.path.join(data_root_dir, train_src_vocab), 
            'data_name': 'src'},
        {'files': os.path.join(data_root_dir, test_src_file),
         'vocab_file': os.path.join(data_root_dir, train_src_vocab),
         'data_name': 'merge'} # though we call this 'merge', actually nothing different with src
    ],
    'batch_size':896,
    'shuffle': False # default=True
}


# # emotional_data
# test_src_file='weibo_senti_100k.txt'
#
# emotional_test_data_params={
#     'datasets':[
#         {'files': os.path.join("data/data_v15_d1g10_transductive_for_base_ori/", 'weibo_senti_100k.txt'),
#             'vocab_file': os.path.join("data/data_v15_d1g10_transductive_for_base_ori/", 'dict.txt'),
#             'data_name': 'src'}
#     ],
#     'batch_size': 64,
#     'shuffle': False # default=True
# }
