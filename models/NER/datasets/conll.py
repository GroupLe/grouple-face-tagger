from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torch import LongTensor
from tqdm.notebook import tqdm
from .tokenization import map_labels_to_wordpiece, tokenize_words
from .reader import read_conll


class ConllDataset(Dataset):
    def __init__(self, sentences):
        self.words_sents = list(map(self._fetch_words, sentences))
        self.label_sents = list(map(self._fetch_labels, sentences))    
        self.labels_n, self.ne2ix = self._get_labels_n(self.label_sents)

    @classmethod
    def from_file(cls, path):
        data = read_conll(path)
        return cls(data)

    def _get_labels_n(self, label_sents):
        labels_types = set()
        for sent in label_sents:
            for label in sent:
                labels_types.add(label)
        ne2ix = {ne: i for i, ne in enumerate(list(labels_types))}
        return len(labels_types), ne2ix
        
    def _fetch_labels(self, sentence):
        return [t[1] for t in sentence]

    def _fetch_words(self, sentence):
        return [t[0] for t in sentence]
    
    def __len__(self):
        return len(self.words_sents)
    
    def __getitem__(self, ix):
        """Returns list of words ixs and labels ids"""
        words_ixs, labels = self.words_sents[ix], self.label_sents[ix]
        return words_ixs, labels
    
    
class ConllCharDataset(ConllDataset):
    """Should implement another train-test split logic"""
    def __init__(self, sentences):
        words_sents = list(map(self._fetch_words, sentences))
        label_sents = list(map(self._fetch_labels, sentences))
        self.labels_n, self.ne2ix = self._get_labels_n(label_sents)
        assert self.labels_n == 2
        
        self.all_words, self.all_labels = self._get_all_wordlabels(words_sents, label_sents)
        self.char2ix = self._get_char2ix_mapper(self.all_words)
        
    def _get_char2ix_mapper(self, all_words):
        all_chars = set()
        for w in all_words:
            for char in w:
                all_chars.add(char)
        char2ix = {c:i for i,c in enumerate(list(all_chars))}
        return char2ix
        
    def _get_all_wordlabels(self, sents, labelslist):
        unique_words = set()
        all_words = []
        all_labels = []
        for sent, labels in zip(sents, labelslist):
            for i in range(len(sent)):
                word, label = sent[i][0], sent[i][-1]
                if word not in unique_words:
                    unique_words.add(word)
                    all_words.append(word)
                    all_labels.append(label == 'PER')
        return all_words, all_labels
        
    def __len__(self):
        return len(self.all_words)
    
    def __getitem__(self, ix):
        """Returns tensor Length x Chars and label"""
        word, label = self.all_words[ix], self.all_labels[ix]
        char_ixs = list(map(lambda c: self.char2ix[c], word))
        char_vecs = LongTensor(one_hot(char_ixs, num_classes=len(self.char2ix)))
        label = LongTensor([label])
        return char_vecs, label
    

class ConllBertDataset(ConllDataset):
    """Conll dataset with bert wordpeace tokenization"""
    def __init__(self, sentences, tokenizer):
        words_sents = list(map(self._fetch_words, sentences))
        label_sents = list(map(self._fetch_labels, sentences))    
        self.tokenizer = tokenizer
        self.words_sents, self.label_sents = self._bertize_wrapper(words_sents, label_sents)
        self.labels_n, self.ne2ix = self._get_labels_n(label_sents)

    def _bertize_wrapper(self, words_sents, label_sents):
        """Applies bertize procedure to given sentences"""
        wp_words_sents = []
        wp_label_sents= []
        
        for words, labels in tqdm(zip(words_sents, label_sents), total=len(words_sents)):
            wp_words, wp_labels = self._bertize(words, labels)
            wp_words_sents.append(wp_words)
            wp_label_sents.append(wp_labels)
        
        return wp_words_sents, wp_label_sents
        
    def _bertize(self, words, labels):
        """Maps word labels to bert wordpiece labels. Returns wordpiece str tokens and wordpiece labels"""
        words = list(map(str.lower, words))
        wp_labels = map_labels_to_wordpiece(words, labels, self.tokenizer)
        wp_words = tokenize_words(words, self.tokenizer)
        return wp_words, wp_labels
    
    @classmethod
    def from_file(cls, path, tokenizer):
        data = read_conll(path)
        return cls(data, tokenizer)
