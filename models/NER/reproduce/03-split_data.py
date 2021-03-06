# Splits data for train-test
from pathlib import Path
from sklearn.model_selection import train_test_split

TEST_PART = 0.2
PATH = Path('../../../data/NER/processed/comments/augmented_10/')

if __name__ == '__main__':
    data = open(PATH / 'raw.txt').read().split('\n\n')
    train, test = train_test_split(data, test_size=TEST_PART, random_state=2021)

    with open(PATH / 'train.txt', 'w') as f:
        f.write('\n\n'.join(train))
    with open(PATH / 'test.txt', 'w') as f:
        f.write('\n\n'.join(test))
