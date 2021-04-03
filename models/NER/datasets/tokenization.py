from transformers import BertTokenizer


def map_labels_to_wordpiece(words: list, labels: list, tokenizer: BertTokenizer):
    """
    Maps labels from original sentence to labels per bert wordpeace token
    @param words:                 words
    @param labels:                labels per word
    @param wordpiece_tokenizer:
    """
    assert len(words) == len(labels)
    wp_labels = []
    
    for word, label in zip(words, labels):
        wp_labels += [label]*len(tokenizer.tokenize(word))
    wp_labels = ['O'] + wp_labels + ['O']
    
    return wp_labels


def tokenize_words(words, tokenizer):
    return tokenizer.encode(' '.join(words))