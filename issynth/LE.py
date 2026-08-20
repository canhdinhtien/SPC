import random
from keytotext import pipeline
import pickle
from util_data import SUBSET_NAMES
from utils import get_dataset_name_for_template

nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")

def word2sentence(dataset, num=200, save_path=''):
    classnames = SUBSET_NAMES[dataset]
    sentence_dict = {}
    for n in classnames:
        sentence_dict[n] = []
    for n in classnames:
        for i in range(num+50):
            sentence = nlp([n], num_return_sequences=1, do_sample=True)
            sentence_dict[n].append(sentence)

    # remove duplicate
    sampled_dict = {}
    for k, v in sentence_dict.items():
        v_unique = list(set(v))
        sampled_v = random.sample(v_unique, num)
        sampled_dict[k] = sampled_v

    r = open(save_path,"wb")
    pickle.dump(sampled_dict, r)
    r.close()

# if __name__ == "__main__":
#     num = sys.argv[1]
#     save_path = sys.argv[2]
#     word2sentence(dataset, int(num), save_path)