'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def init_embeddings(embeddings) -> None:
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map, output_folder):
    emb_basename = os.path.basename(emb_file)
    cache_path = os.path.join(output_folder, emb_basename + '.pth.tar')

    # no cache, load embeddings from .txt file
    if not os.path.isfile(cache_path):
        # find embedding dimension
        with open(emb_file, 'r') as f:
            embed_dim = len(f.readline().split(' ')) - 1
            num_lines = len(f.readlines())

        vocab = set(word_map.keys())

        # create tensor to hold embeddings, initialize
        embeddings = torch.FloatTensor(len(vocab), embed_dim)
        init_embeddings(embeddings)

        # read embedding file
        for line in tqdm(open(emb_file, 'r'), total=num_lines, desc='Loading embeddings'):
            line = line.split(' ')

            emb_word = line[0]
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            # ignore word if not in train_vocab
            if emb_word not in vocab:
                continue

            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

        # create cache file so we can load it quicker the next time
        print('Saving vectors to {}'.format(cache_path))
        torch.save((embeddings, embed_dim), cache_path)

    # load embeddings from cache
    else:
        print('Loading embeddings from {}'.format(cache_path))
        embeddings, embed_dim = torch.load(cache_path)

    return embeddings, embed_dim


def load_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    model = checkpoint['model']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    dataset_name = checkpoint['dataset_name']
    word_map = checkpoint['word_map']
    start_epoch = checkpoint['epoch'] + 1

    return model, model_name, optimizer, dataset_name, word_map, start_epoch


def save_checkpoint(epoch, model, model_name, optimizer, dataset_name, word_map, checkpoint_path, checkpoint_basename='checkpoint'):
    state = {
        'epoch': epoch,
        'model': model,
        'model_name': model_name,
        'optimizer': optimizer,
        'dataset_name': dataset_name,
        'word_map': word_map
    }
    save_path = os.path.join(checkpoint_path, checkpoint_basename + '.pth.tar')
    torch.save(state, save_path)
