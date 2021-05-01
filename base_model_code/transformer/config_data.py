import os
max_decoding_length = 200
num_epochs=20
display=16


# vocab_length=12009
# vocab_length+=3 # BOS,EOS,OOV

# train
train_data_params={
    'datasets':[
        {'files': 'data/train400w.src',
            'vocab_file': 'data/dict400w',
            'data_name': 'src'},
        {'files': 'data/train400w.tgt',
            'vocab_file': 'data/dict400w',
            'data_name': 'tgt'},
    ],
    'batch_size': 64
    # 'shuffle': False # default=True
}


# test
test_data_params={
    'datasets':[
        {'files': 'data/test400w.src',
            'vocab_file': 'data/dict400w',
            'data_name': 'src'},
    ],
    'batch_size':896,
    'shuffle': False # default=True
}


