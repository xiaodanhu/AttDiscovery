import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

''' set random seeds '''
seed_val = 312
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


''' set up device '''
# use cuda
if torch.cuda.is_available():  
    dev = "cuda:3"  
else:  
    dev = "cpu"
# CUDA_VISIBLE_DEVICES=0,1,2,3  
device = torch.device(dev)
print(f'[INFO]using device {dev}')


### preliminary ###
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "clip/bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


### preprocess LAD data ###
def process_LAD_to_pickle():
    file_root = '/shared/nas/data/m1/wangz3/multimedia_attribute/data/LAD'
    data_root = file_root + '/LAD_annotations/'
    img_root = file_root + '/LAD_images/'

    
    # load attributes list
    attributes_list_path = data_root + 'attribute_list.txt'
    fsplit = open(attributes_list_path, 'r', encoding='UTF-8')
    lines_attribute = fsplit.readlines()
    fsplit.close()
    list_attribute = list()
    list_attribute_value = list()
    for each in lines_attribute:
        tokens = each.split(', ')
        list_attribute.append(tokens[0])
        list_attribute_value.append(tokens[1])

    # load label list
    label_list_path = data_root + 'label_list.txt'
    fsplit = open(label_list_path, 'r', encoding='UTF-8')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_label = dict()
    list_label_value = list()
    for each in lines_label:
        tokens = each.split(', ')
        list_label[tokens[0]]=tokens[1]
        list_label_value.append(tokens[1])

    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from PIL import Image

    preprocess = Compose([
        Resize((224, 224), interpolation=Image.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor()
    ])

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

    # load all the labels, attributes, images data from the LAD dataset
    attributes_per_class_path = data_root + 'attributes.txt'
    fattr = open(attributes_per_class_path, 'r', encoding='UTF-8')
    lines_attr = fattr.readlines()
    fattr.close()
    images = list()
    attr = list()
    labels = list()
    for each in lines_attr:
        tokens = each.split(', ')
        labels.append(list_label[tokens[0]])
        img_path = tokens[1]
        image = preprocess(Image.open(os.path.join(img_root, img_path)).convert("RGB"))
        images.append(image)
        attr_r = list(map(int, tokens[2].split()[1:-1]))
        attr.append([val for i,val in enumerate(list_attribute_value) if attr_r[i] == 1])
    
    # Dump processed image and text to local
    with open('../checkpoints/data_img_raw.pkl', 'wb') as file:
        pickle.dump(images, file)
    with open('../checkpoints/data_txt_raw.pkl', 'wb') as file:
        pickle.dump({'label': labels, 'att': attr}, file)

def get_LAD_img_text_tensor_inputs(images, labels, attr):        

    # normalize images
    image_input = torch.tensor(np.stack(images)).to(device)
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    # Convert labels to tokens
    tokenizer_label = SimpleTokenizer()
    text_tokens = [tokenizer_label.encode(desc) for desc in labels]

    sot_token = tokenizer_label.encoder['<|startoftext|>']
    eot_token = tokenizer_label.encoder['<|endoftext|>']

    text_inputs_label = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        text_inputs_label[i, :len(tokens)] = torch.tensor(tokens)
    text_inputs_label = text_inputs_label.to(device)

    
    # Convert attributes to tokens
    tokenizer_att = SimpleTokenizer()
    text_tokens = [[tokenizer_att.encode(desc) for desc in att] for att in attr]

    sot_token = tokenizer_att.encoder['<|startoftext|>']
    eot_token = tokenizer_att.encoder['<|endoftext|>']
    text_inputs_att = list()

    for j, tokens_img in enumerate(text_tokens):
        text_input = torch.zeros(len(tokens_img), model.context_length, dtype=torch.long)
        for i, tokens in enumerate(tokens_img):
            tokens = [sot_token] + tokens + [eot_token]
            text_input[i, :len(tokens)] = torch.tensor(tokens)
        text_inputs_att.append(text_input.to(device))

    return image_input, text_inputs_label, text_inputs_att

def get_LAD_features_CLIP():
    '''model'''
    model = torch.jit.load("../checkpoints/model.pt").to(device).eval()
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    
    '''load LAD pickle'''
    with open('../checkpoints/data_img_raw.pkl', 'rb') as file:
        images = pickle.load(file)
    with open('../checkpoints/data_txt_raw.pkl', 'rb') as file:
        b = pickle.load(file)
    
    labels = b['label']
    attr = b['att']

    image_input, text_inputs_label, text_inputs_att = get_LAD_img_text_tensor_inputs(images, labels, attr)

    # get image features
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    # get label features    
    with torch.no_grad():
        label_fea = model.encode_text(text_inputs_label.to(device)).float()

    # get attr features    
    with torch.no_grad():
        text_feature = list()
        for txt in text_inputs_att:
            if len(txt) == 0:
                text_feature.append(torch.empty(0, 512).to(device))
            else:
                text_feature.append(model.encode_text(txt).float())

    image_features /= image_features.norm(dim=-1, keepdim=True)

    label_fea /= label_fea.norm(dim=-1, keepdim=True)

    text_feature = torch.stack([torch.mean(item,0) for item in text_feature])
    text_feature /= text_feature.norm(dim=-1, keepdim=True)

    # Save image and text features
    with open('../checkpoints/data_txt_feature.pkl', 'wb') as file:
        pickle.dump({'label': label_fea, 'att': text_feature}, file)
    with open('../checkpoints/data_img_feature.pkl', 'wb') as file:
        pickle.dump(image_features, file)


"""Dataloader for classification using precomputed fix length features from CLIP"""
from torch.utils.data import Dataset
from sklearn import preprocessing

class LAD_Feature_Dataset(Dataset):

    def __init__(self, image_features, text_feature, labels, data_indx):
        self.image_features = image_features
        self.text_feature = text_feature
        self.labels = labels
        self.data_indx = data_indx
        # self.imgs = image_input
        # self.attr = attr

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': self.image_features[idx], 
                  'attribute': self.text_feature[idx], 
                  'label': self.labels[idx],
                  'data_indx': self.data_indx[idx]
                #   'imgs': self.imgs[idx],
                #   'attr': self.attr[idx]
                 }
        return sample

def get_LAD_features_dataloader():
    # load text feature
    with open('../checkpoints/data_txt_feature.pkl', 'rb') as file:
        b = pickle.load(file)

    label_fea = b['label']
    text_feature = b['att']

    # load img feature
    with open('../checkpoints/data_img_feature.pkl', 'rb') as file:
        image_features = pickle.load(file)

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    class_list = list(le.classes_)
    labels_list = torch.tensor(le.transform(labels)).to(device)

    attr_ = [';'.join(attr[0]) for item in attr]
    data_indx = list(range(4600))
    # dataset = LAD_Feature_Dataset(image_features, text_feature, labels_list, torch.tensor(np.stack(images)).to(device), attr_)
    dataset = LAD_Feature_Dataset(image_features, text_feature, labels_list, data_indx)
    train_set, test_set = torch.utils.data.random_split(dataset,[4600-500,500])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


"""Dataloader for classification from raw LAD image and text"""
class LAD_raw_Dataset(Dataset):
    # TODO: load LAD img and text
    def __init__(self, image_input, text_inputs_label, text_inputs_attr, labels, data_indx):
        self.image_input = image_input
        self.text_inputs_label = text_inputs_label
        self.text_inputs_attr = text_inputs_attr
        self.labels = labels
        self.data_indx = data_indx

    def __len__(self):
        return len(self.image_input)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': self.image_input[idx], 
                  'text_label': self.text_inputs_label[idx], 
                  'text_attr': self.text_inputs_attr[idx], # a list of tensors #TODO: how to use it?
                  'label': self.labels[idx],
                  'data_indx': self.data_indx[idx]
                 }
        return sample

def get_LAD_features_dataloader():
    '''load LAD pickle'''
    with open('../checkpoints/data_img_raw.pkl', 'rb') as file:
        images = pickle.load(file)
    with open('../checkpoints/data_txt_raw.pkl', 'rb') as file:
        b = pickle.load(file)
        
    labels = b['label']
    attr = b['att']

    image_input, text_inputs_label, text_inputs_att = get_LAD_img_text_tensor_inputs(images, labels, attr)

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    class_list = list(le.classes_)
    labels_list = torch.tensor(le.transform(labels)).to(device)

    # attr_ = [';'.join(attr[0]) for item in attr]
    data_indx = list(range(4600))

    dataset = LAD_raw_Dataset(image_input, text_inputs_label, text_inputs_att, labels_list, data_indx)

    train_set, test_set = torch.utils.data.random_split(dataset,[4600-500,500])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    return trainloader, testloader


if __name__ == '__main__':
    process_LAD_to_pickle()