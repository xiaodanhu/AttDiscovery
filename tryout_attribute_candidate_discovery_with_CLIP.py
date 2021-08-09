import torch
import clip
from PIL import Image
import spacy
import numpy as np

def get_attr_prompts(class_name, noun_chunk):
    templates = [f'a {class_name} with {noun_chunk}']
    # templates = [f'a {class_name} featured {noun_chunk}']
    # templates = [f'a {class_name} with {noun_chunk}', f'a {class_name} featured {noun_chunk}', f'a {class_name} showing {noun_chunk}']
    return templates

def get_attribute_spans(image, text_raw, class_name, nlp, k = 3):
    doc = nlp(text_raw)
    prompts = [] # a list of lists, each item is a list of filled prompt templete for each noun_chunk
    noun_chunk_list = []
    for chunk in doc.noun_chunks:
        # pos_list = [token.pos_ for token in chunk]
        ## filter out noun chunks without adj
        # if 'ADJ' in pos_list:
        noun_chunk_list.append(chunk.text)
        prompts.append(get_attr_prompts(class_name, chunk.text))
    
    queries = []
    for template_i in range(len(prompts[0])):
        query = []
        for noun_chunk_i in range(len(prompts)):
            query.append(prompts[noun_chunk_i][template_i])
        queries.append(query)

    scores = None
    for query in queries:
        print("query:", query)
        text = clip.tokenize(query).to(device) # query is a list of filled templetes of each candidate noun_chunk, i.e ['a <dog> with <white paws>','a <dog> with <The breed>'...]

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if scores is None:
            scores = probs
        else:
            scores = np.concatenate((scores, probs), axis=0)
        print("scores:", scores)

    #TODO: average over various template scores:
    # now we only use one template:

    final_scores = scores[0]
    ranked = np.argsort(-1*final_scores)[:k]
    top_k_candidates = [(noun_chunk_list[j],final_scores[j]) for j in ranked]
    print(f'top {k} candidates:', top_k_candidates)

    



if __name__ == '__main__':

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    """specify class"""
    # class_name = 'Siberian Husky'
    # class_name = 'Persian cat'
    class_name = 'German_Shepherd'

    """load image"""
    # image = preprocess(Image.open("imgs/Husky_L.jpg")).unsqueeze(0).to(device)
    # image = preprocess(Image.open("imgs/White_Persian_Cat.jpg")).unsqueeze(0).to(device)
    image = preprocess(Image.open("imgs/German_Shepherd.jpg")).unsqueeze(0).to(device)

    """load original text"""
    # text_raw = 'The Siberian Husky is a medium-sized working sled dog breed. The breed belongs to the Spitz genetic family. It is recognizable by its thickly furred double coat, erect triangular ears, and distinctive markings, and is smaller than the similar-looking Alaskan Malamute.'
    # text_raw = 'The Persian cat is a long-haired breed of cat characterized by its round face and short muzzle. It is also known as the \"Persian Longhair\" in the English-speaking countries. The first documented ancestors of the Persian were imported into Italy from Persia around 1620. Recognized by the cat fancy since the late 19th century, it was developed first by the English, and then mainly by American breeders after the Second World War. Some cat fancier organizations\' breed standards subsume the Himalayan and Exotic Shorthair as variants of this breed, while others treat them as separate breeds.'
    text_raw = 'They have a domed forehead, a long square-cut muzzle with strong jaws and a black nose. The eyes are medium-sized and brown. The ears are large and stand erect, open at the front and parallel, but they often are pulled back during movement. A German Shepherd has a long neck, which is raised when excited and lowered when moving at a fast pace as well as stalking. The tail is bushy and reaches to the hock.'
    
    """set up nlp"""
    nlp = spacy.load("en_core_web_sm")

    get_attribute_spans(image, text_raw, class_name, nlp, k = 5)

    




