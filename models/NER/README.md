# Persons names recognition

## Reproduce
1. Collect data
2. Use `reproduce/manual_data_markup.ipynb` to create conll-kind dataset
3. Use `reproduce/augment_comments.ipynb` to augment text data with extra names. Here x30 augmentation used
4. Train RuBERT conversational based model for NER with `train_rubert.ipynb`

## Usage

See `eval_model.ipynb`


## Known issues

- works bad on eng text. Detects everything as name
