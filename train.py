#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch as T
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import math

from tqdm import tqdm

from helpers import *
from model import *
from generate import *
from util import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--train_file', type=str, default="./train.txt")
argparser.add_argument('--eval_file', type=str, default="./eval.txt")
argparser.add_argument('--rnn_type', type=str, default="dnc")
argparser.add_argument('--n_epochs', type=int, default=1000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=200)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=256)
argparser.add_argument('--cuda', type=int, default=0)
argparser.add_argument('--nr_cells', type=int, default=32)
argparser.add_argument('--read_heads', type=int, default=4)
argparser.add_argument('--cell_size', type=int, default=32)
argparser.add_argument('--reset_experience', type=str, default='0')
args = argparser.parse_args()

if args.cuda != -1:
  print("Using GPU ", args.cuda)


train_file, train_len = read_file(args.train_file)
eval_file, eval_len = read_file(args.eval_file)


def random_training_set(chunk_len, batch_size, file, file_len):
  inp = []  # T.LongTensor(batch_size, chunk_len)
  target = []  # T.LongTensor(batch_size, chunk_len)

  for bi in range(batch_size):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp.append(char_tensor(chunk[:-1]))
    target.append(char_tensor(chunk[1:]))

  inp = cuda(T.stack(inp, 0), gpu_id=args.cuda)
  target = cuda(T.stack(target, 0), gpu_id=args.cuda)
  return inp, target


def eval(inp, target):
  hidden = None
  loss = 0

  lm.eval()

  for c in range(args.chunk_len):
    output, hidden = lm(inp[:, c], hidden)
    loss += criterion(output.view(args.batch_size, -1), target[:, c])

  lm.train()
  return loss.data[0] / args.chunk_len


def train(inp, target):
  hidden = None
  lm_optimizer.zero_grad()
  loss = 0

  for c in range(args.chunk_len):
    output, hidden = lm(inp[:, c], hidden)
    loss += criterion(output.view(args.batch_size, -1), target[:, c])

  loss.backward()
  nn.utils.clip_grad_norm(lm.parameters(), 5.0)
  lm_optimizer.step()

  return loss.data[0] / args.chunk_len


def save():
  save_filename = os.path.splitext(os.path.basename(args.train_file))[0] + '.pt'
  T.save(lm, save_filename)
  print('Saved as %s' % save_filename)

# Initialize models and start training

if args.rnn_type == 'dnc':
  lm = CharLM(
      n_characters,
      args.hidden_size,
      n_characters,
      rnn_type=args.rnn_type,
      n_layers=args.n_layers,
      nr_cells=args.nr_cells,
      read_heads=args.read_heads,
      cell_size=args.cell_size,
      gpu_id=args.cuda,
      reset_experience=(args.reset_experience == '1')
  )
else:
  lm = CharLM(
      n_characters,
      args.hidden_size,
      n_characters,
      rnn_type=args.rnn_type,
      n_layers=args.n_layers,
  )
lm_optimizer = T.optim.Adam(lm.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda != -1:
  lm.cuda(args.cuda)

start = time.time()
all_losses = []
loss_avg = 0

try:
  print("Training for %d epochs..." % args.n_epochs)
  print('With arguments', args)
  for epoch in tqdm(range(1, args.n_epochs + 1)):
    # batches = train_len / (args.batch_size * args.chunk_len)
    # for batch in range(int(batches)):
    loss = train(*random_training_set(args.chunk_len, args.batch_size, train_file, train_len))
    loss_avg += loss

    if epoch % args.print_every == 0:
      eval_loss = eval(*random_training_set(args.chunk_len, args.batch_size, eval_file, eval_len))

      print('[%s (%d %d%%) with loss %.4f, eval log-perplexity %f]' %
            (time_since(start), epoch, epoch / args.n_epochs * 100, loss, eval_loss))
      print(generate(lm, 'Th', 100, cuda=args.cuda), '\n')

  print("Saving...")
  save()

except KeyboardInterrupt:
  print("Saving before quit...")
  save()
