import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

import pickle

import numpy as np

import time

import json

import utils

from models.HAN import HAN


class DocDataset(Dataset):
    train_file = 'train_batch'
    val_file = 'val_batch'
    test_file = 'test_batch'

    def __init__(self, holdoutroot, mode='train') -> None:

        self.mode = mode  # training set or test set

        assert mode in {'train', 'test', 'val'}

        if self.mode == 'train':
            file_name = self.train_file
        elif self.mode == 'val':
            file_name = self.val_file
        else:
            file_name = self.test_file

        # load data
        file_path = os.path.join(holdoutroot, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['labels']
            self.holdout = entry['holdout']

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), \
            torch.LongTensor([self.data['sentences_per_document'][i]]), \
            torch.LongTensor(self.data['words_per_sentence'][i]), \
            torch.LongTensor([self.targets[i]])

    def __len__(self) -> int:
        return len(self.targets)


def load_data(output_path, split, build_vocab=True):
    assert split in {'train', 'test', 'val'}

    # test or val
    if split == 'test' or split == 'val':
        test_loader = DataLoader(
            DocDataset(output_path, split),
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        return test_loader
    # train
    else:
        # dataloaders
        train_loader = DataLoader(
            DocDataset(output_path, 'train'),
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        if build_vocab == False:
            return train_loader

        else:
            # load word2ix map
            with open(os.path.join(output_path, 'word_map.json'), 'r') as j:
                word_map = json.load(j)
            # size of vocabulary
            vocab_size = len(word_map)

            # word embeddings
            '''
            if emb_pretrain == True:
                # load Glove as pre-trained word embeddings for words in the word map
                emb_path = os.path.join(emb_folder, emb_filename)
                embeddings, emb_size = utils.load_embeddings(
                    emb_file=os.path.join(emb_folder, emb_filename),
                    word_map=word_map,
                    output_folder=output_path
                )
            
            # or initialize embedding weights randomly
            else:
            '''
            embeddings = None
            emb_size = 256

            return train_loader, embeddings, emb_size, word_map, 10, vocab_size


def check_done(path):
    if os.path.isfile(path):
        if os.stat(path).st_size > 0:
            return True
    return False


def adjust_learning_rate(optimizer: optim.Optimizer, scale_factor: float) -> None:
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def train_model(run, data_folder, result_folder, mode, holdout_class, a):
    saving_dir = os.path.join(result_folder, 'amazon', 'amazon-%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), a))
    os.makedirs(saving_dir, exist_ok=True)
    saving_file = os.path.join(saving_dir, 'model_%d.pth' % run)

    if check_done(saving_file):
        print("Already trained for run %d with holdout class %s and a %d. Skip to the next run." % (run, holdout_class, a))
        return

    lr = 0.001
    lr_decay = 0.3

    EPOCH = 5
    #EPOCH = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('Preparing data..')
    tic = time.perf_counter()

    holdoutroot = os.path.join(data_folder, 'amazon', '%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), a))

    # load data
    train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(holdoutroot, 'train', True)

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    # Model
    print('Building model..')
    tic = time.perf_counter()

    net = HAN(
        n_classes=n_classes,
        vocab_size=vocab_size,
        embeddings=embeddings,
        emb_size=emb_size,
        fine_tune=True,
        word_rnn_size=50,
        sentence_rnn_size=50,
        word_rnn_layers=1,
        sentence_rnn_layers=1,
        word_att_size=100,
        sentence_att_size=100,
        dropout=0.3
    )

    # loss functions
    criterion = nn.CrossEntropyLoss()

    net = net.to(device)
    criterion = criterion.to(device)

    checkpoint_file = 'ckpt_%s_%s_%d_%d.pth' % (mode, holdout_class, a, run)
    if os.path.isfile(checkpoint_file):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch'] + 1
    else:
        optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr
        )

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    # Training

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for i, batch in enumerate(train_loader):
            documents, sentences_per_document, words_per_sentence, labels = batch

            documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
            sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(device)  # (batch_size)

            # forward
            scores, _, _ = net(
                documents,
                sentences_per_document,
                words_per_sentence
            )  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

            # calc loss
            loss = criterion(scores, labels)  # scalar

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # find accuracy
            total += labels.size(0)
            _, predictions = scores.max(dim=1)  # (n_documents)
            correct += torch.eq(predictions, labels).sum().item()

            # keep track of metrics
            train_loss += loss.item()

        # print training status
        if epoch % 1 == 0:
            print("Loss: %.3f | Acc: %.3f%% (%d/%d)\n" %
                  (train_loss/len(train_loader), 100.*correct/total, correct, total))

    print('Training...')
    tic = time.perf_counter()

    # epochs
    for epoch in range(start_epoch, EPOCH):
        print('Epoch: %d' % epoch)

        # trian an epoch
        train(epoch)

        if epoch % 1 == 0:
            print('Saving check point ..')
            sav_tic = time.perf_counter()

            state = {
                'net': net.state_dict(),
                'optimizer': optimizer,
                'epoch': epoch
            }
            torch.save(state, 'ckpt_%s_%s_%d_%d.pth' % (mode, holdout_class, a, run))

            sav_toc = time.perf_counter()
            print("Done in %f seconds" % (sav_toc - sav_tic))

        # decay learning rate every epoch
        adjust_learning_rate(optimizer, lr_decay)

    toc = time.perf_counter()
    print("Training done in %f seconds" % (toc - tic))

    print('Saving final model')
    tic = time.perf_counter()
    torch.save(net.state_dict(), saving_file)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print('Deleting check point')
    tic = time.perf_counter()
    os.remove('ckpt_%s_%s_%d_%d.pth' % (mode, holdout_class, a, run))
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))


def test_model(run, data_folder, result_folder, mode, holdout_class, a):
    saving_dir = os.path.join(result_folder, 'amazon', 'amazon-%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), a))
    model_saving_file = os.path.join(saving_dir, 'model_%d.pth' % run)
    outputs_saving_file = os.path.join(saving_dir, 'outputs_%d' % run)

    if check_done(outputs_saving_file):
        print("Already evaluate for run %d with holdout class %s and a %d. Skip to the next evaluation run." % (run, holdout_class, a))
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('Preparing data..')
    tic = time.perf_counter()

    holdoutroot = os.path.join(data_folder, 'amazon', '%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), a))

    train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(holdoutroot, 'train', True)
    val_loader = load_data(holdoutroot, 'val')
    test_loader = load_data(holdoutroot, 'test')

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print('Loading model')
    tic = time.perf_counter()

    # load model
    net = HAN(
        n_classes=n_classes,
        vocab_size=vocab_size,
        embeddings=embeddings,
        emb_size=emb_size,
        fine_tune=True,
        word_rnn_size=50,
        sentence_rnn_size=50,
        word_rnn_layers=1,
        sentence_rnn_layers=1,
        word_att_size=100,
        sentence_att_size=100,
        dropout=0.3
    )

    state_dict = torch.load(model_saving_file)
    net.load_state_dict(state_dict)

    net = net.to(device)
    net.eval()

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    def evaluate(data_loader):

        outputs_list = []
        targets_list = []

        correct = 0
        total = 0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                documents, sentences_per_document, words_per_sentence, labels = batch

                documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
                sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
                words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
                labels = labels.squeeze(1).to(device)  # (batch_size)

                # forward
                scores, _, _ = net(
                    documents,
                    sentences_per_document,
                    words_per_sentence
                )  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

                # accuracy
                total += labels.size(0)
                _, predictions = scores.max(dim=1)  # (n_documents)
                correct += torch.eq(predictions, labels).sum().item()

                outputs_list.append(scores.cpu().numpy())
                targets_list.extend(labels.cpu())
            outputs_list = np.vstack(outputs_list)

            print("Acc: %.3f%% (%d/%d)\n" % (100.*correct/total, correct, total))

        return outputs_list, targets_list

    results = {}

    print("Evaluating training data")
    tic = time.perf_counter()
    results['train_outputs'], results['train_targets'] = evaluate(train_loader)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print("Evaluating validation data")
    tic = time.perf_counter()
    results['val_outputs'], results['val_targets'] = evaluate(val_loader)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print("Evaluating test data")
    tic = time.perf_counter()
    results['test_outputs'], results['test_targets'] = evaluate(test_loader)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print("Saving output file")
    tic = time.perf_counter()
    torch.save(results, outputs_saving_file)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))


def main():
    parser = argparse.ArgumentParser(description='Run experiment with holdout AMAZON and HAN')
    parser.add_argument('data_folder', help='data folder')
    parser.add_argument('result_folder', help='result folder')
    parser.add_argument('mode', choices=['holdout'], help='the mode')
    parser.add_argument('holdout_class', help='the holdout class')
    parser.add_argument('ratio', help='the ratio of holdout')
    parser.add_argument('run_id', help='the id of the run')

    args = parser.parse_args()

    data_folder = args.data_folder
    result_folder = args.result_folder
    mode = args.mode

    run_id = int(args.run_id)
    holdout_class = args.holdout_class
    ratio = int(args.ratio)

    train_model(run_id, data_folder, result_folder, mode, holdout_class, ratio)
    test_model(run_id, data_folder, result_folder, mode, holdout_class, ratio)


if __name__ == "__main__":
    main()
