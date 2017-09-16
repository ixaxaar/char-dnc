# char-rnn.pytorch

Char RNN based neural language model which supports GRUs, LSTMs and DNCs as RNN units.

## Training

Download [this Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (from the original char-rnn) as `shakespeare.txt`.  Or bring your own dataset &mdash; it should be a plain text file (preferably ASCII).

Run `train.py` with the dataset filename to train and save the network:

```
> python train.py shakespeare.txt

Training for 2000 epochs...
(... 10 minutes later ...)
Saved as shakespeare.pt
```
After training the model will be saved as `[filename].pt`.

### Training options

```
Usage: train.py [filename] [options]

Options:
--model            Whether to use LSTM, GRU or DNC units    gru
--n_epochs         Number of epochs to train           1000
--print_every      Log learning rate at this interval  100
--hidden_size      Hidden size of DNC                  200
--n_layers         Number of DNC layers                2
--learning_rate    Learning rate                       0.001
--chunk_len        Length of training chunks           200
--batch_size       Number of examples per batch        256
--cuda             GPU ID to use, -1 for cpu
--nr_cells         Number of cells of the DNC          32
--read_heads       Number of read heads                4
--cell_size        Size of each DNC cell               32
--reset_experience Should we reset DNC values per minibatch? False
```

## Generation

Run `generate.py` with the saved model from training, and a "priming string" to start the text with.

```
> python generate.py shakespeare.pt --prime_str "Where"

Where, you, and if to our with his drid's
Weasteria nobrand this by then.

AUTENES:
It his zersit at he
```

### Generation options
```
Usage: generate.py [filename] [options]

Options:
-p, --prime_str      String to prime generation with
-l, --predict_len    Length of prediction
-t, --temperature    Temperature (higher is more chaotic)
--cuda               Use CUDA
```

Copied and modified form a PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) for character-level text generation, which is copied from [the Practical PyTorch series](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb).

