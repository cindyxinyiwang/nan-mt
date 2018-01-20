# Non-autoregressive Neural Machine Translation (NaN MT)

## Data pre-processing
First, download IWSLT 2016 en-de data from
```
https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=en&tlang=de
```

Unzip the data into
```
./data/en-de
```

Strip the XML tags from IWSLT data
```bash
python src/pre_process.py
```

Next, download the SentencePiece code:
```
https://github.com/google/sentencepiece
```

Train a BPE model and generate the BPE tokenized texts:
```bash
./scripts/pre_process_bpe.sh
```
