import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data import breakfast as breakfast


def _display_distribution(labels, n_labels):
    for label in labels:
        assert label < n_labels, 'some labels are not smaller than {}'.format(n_labels)

    fig = plt.hist(x=labels, bins=n_labels)
    plt.title('Test Label Distribution')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == '__main__':
    _, labels, logits = breakfast.get_data(split='test')

    logits = [logit for logit in logits if logit]
    logits = [logit for logit in logits if logit < 48]
    # todo: uncomment to view sorted version
    # logit_counter = Counter(logits)
    # most_common = logit_counter.most_common()
    # remap = {}
    # for i in range(len(most_common)):
    #     mc_tuple = most_common[i]
    #     remap[mc_tuple[0]] = i
    # logits = [remap[logit] for logit in logits]

    logits = np.array(logits).reshape(-1)
    _display_distribution(logits, n_labels=max(logits) + 1)

