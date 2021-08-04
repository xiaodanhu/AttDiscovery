import os
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from PIL import Image
import requests

from model.modeling_VL_transformer import VLTransformerBert, VLTransformerBertForClassification 

def tryout():
    # get image features
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    img_max_len = 200

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    img_features = last_hidden_states[0] # (sequence_len, img_dim) <- i.e. (197,768) 
    
    print('original img features size:',img_features.size())
    # truncate/pad image feature sequence to fix length
    img_len = img_features.shape[0]
    if img_len > img_max_len:
        img_features = img_features[0 : img_max_len,]
        img_len = img_max_len
    else:
        padding_matrix = torch.zeros((img_max_len - img_len,
                                        img_features.shape[1]))
        img_features = torch.cat((img_features, padding_matrix), 0)
    print('padded img features size:',img_features.size())
    img_attention_mask = torch.zeros(img_max_len)
    img_attention_mask[0 : img_len] = torch.ones(img_len)
    ''' mask value:
        1 for tokens that are not masked,
        0 for tokens that are masked.
    '''
    print('img attention mask:',img_attention_mask)


    # text
    text = 'There are two little cats in the image.'
    bert_config_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_config_name)
    config = BertConfig.from_pretrained(bert_config_name)
    # add img config
    config.img_dim = 768
    config.hidden_dropout_prob = 0.1

    text_max_len = 128
    inputs = tokenizer(text, padding='max_length', max_length=text_max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    input_ids = inputs['input_ids'][0]
    text_att_mask = inputs['attention_mask'][0]

    print('text input size:',input_ids.size())
    print('text attention mask:', text_att_mask)
    
    concat_attention_mask = torch.cat((text_att_mask, img_attention_mask))
    print('concat_attention_mask:', concat_attention_mask)
    

    # add batch dimension
    input_ids_b = torch.unsqueeze(input_ids,0)
    concat_attention_mask_b = torch.unsqueeze(concat_attention_mask,0)
    img_features_b = torch.unsqueeze(img_features,0)
    assert input_ids_b.size()[1] + img_features_b.size()[1] == concat_attention_mask_b.size()[1]
    # model:
    VLT = VLTransformerBert(config, bert_config_name = 'bert-base-uncased')
    output = VLT(
        input_ids=input_ids_b,
        attention_mask=concat_attention_mask_b,
        return_dict=True,
        img_features = img_features_b,
    )
    print('output last hidden states size:', output.last_hidden_state.size())
    print()

    # classification
    config.num_labels = 3
    label = torch.tensor(1, dtype = torch.long)
    label_b = torch.unsqueeze(label, 0)
    VLT_classification = VLTransformerBertForClassification(config, bert_config_name)
    output = VLT_classification(
        input_ids=input_ids_b,
        attention_mask=concat_attention_mask_b,
        return_dict=True,
        img_features = img_features_b,
        labels = label_b,
    )
    print('output classification logits:',output.logits)
    print('output classification loss:',output.loss)

if __name__ == '__main__':
    tryout()

