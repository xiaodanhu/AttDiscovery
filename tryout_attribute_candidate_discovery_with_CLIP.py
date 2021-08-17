import torch
import clip
from PIL import Image
import spacy
import numpy as np
import os, json, cv2
from tryout_detectron2 import get_region_proposals, get_detectron_inputs

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_attr_prompts(class_name, noun_chunk):
    templates = [f'a {class_name} with {noun_chunk}']
    # templates = [f'a {class_name} featured {noun_chunk}']
    # templates = [f'a {class_name} with {noun_chunk}', f'a {class_name} featured {noun_chunk}', f'a {class_name} showing {noun_chunk}']
    return templates

def get_attribute_candidate_prompts(text_raw, nlp, if_use_template = False, class_name = None, method = 'DependencyParser'):
    """
        method: AMR, DependencyParser
    """
    doc = nlp(text_raw)
    prompts = [] # a list of lists, each item is a list of filled prompt templete for each noun_chunk
    if method == 'DependencyParser':
        noun_chunk_list = []
        for chunk in doc.noun_chunks:
            # pos_list = [token.pos_ for token in chunk]
            ## filter out noun chunks without adj
            # if 'ADJ' in pos_list:
            noun_chunk_list.append(chunk.text)
            if if_use_template and class_name:
                prompts.append(get_attr_prompts(class_name, chunk.text)[0]) # NOTE: only use one template for now
            else:
                prompts.append(chunk.text)
    return noun_chunk_list, prompts

def get_probs(model, preprocess, noun_chunk_list, prompts, image_patches, device):
    # NOTE: only use one template/the text itself for each noun_chunk
    
    # prepare image inputs
    image_tensors = []
    for patch in image_patches:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = Image.fromarray(patch) # convert cv2 -> PIL image
        image_tensors.append(preprocess(patch).to(device))
    image_inputs = torch.stack(image_tensors).to(device)
    print('image inputs size:',image_inputs.size())
    # prepare text inputs
    text_inputs = clip.tokenize(prompts).to(device)
    print('text inputs size:',text_inputs.size())

    # get probs
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        
        logits_per_image, logits_per_text = model(image_inputs, text_inputs)
        print('logits_per_image size:',logits_per_image.size())
        print('logits_per_text size:',logits_per_text.size())
        probs_per_image = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs_per_text = logits_per_text.softmax(dim=-1).cpu().numpy()

    return probs_per_image, probs_per_text


def get_top_k_attribute_spans_with_whole_image(image, text_raw, class_name, nlp, k, if_use_template = True):

    noun_chunk_list, prompts = get_attribute_candidate_prompts(text_raw, nlp, if_use_template, class_name) 
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
    return noun_chunk_list, scores

    
def main():

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    """specify class"""
    class_name = 'Siberian Husky'
    # class_name = 'Persian cat'
    # class_name = 'German_Shepherd'

    """load image"""
    image = preprocess(Image.open("test_images/Husky_L.jpg")).unsqueeze(0).to(device)
    # image = preprocess(Image.open("test_images/White_Persian_Cat.jpg")).unsqueeze(0).to(device)
    # image = preprocess(Image.open("test_images/German_Shepherd.jpg")).unsqueeze(0).to(device)
    print('image size:',image.size())
    
    """load original text"""
    text_raw = 'The Siberian Husky is a medium-sized working sled dog breed. The breed belongs to the Spitz genetic family. It is recognizable by its thickly furred double coat, erect triangular ears, and distinctive markings, and is smaller than the similar-looking Alaskan Malamute.'
    # text_raw = 'The Persian cat is a long-haired breed of cat characterized by its round face and short muzzle. It is also known as the \"Persian Longhair\" in the English-speaking countries. The first documented ancestors of the Persian were imported into Italy from Persia around 1620. Recognized by the cat fancy since the late 19th century, it was developed first by the English, and then mainly by American breeders after the Second World War. Some cat fancier organizations\' breed standards subsume the Himalayan and Exotic Shorthair as variants of this breed, while others treat them as separate breeds.'
    # text_raw = 'They have a domed forehead, a long square-cut muzzle with strong jaws and a black nose. The eyes are medium-sized and brown. The ears are large and stand erect, open at the front and parallel, but they often are pulled back during movement. A German Shepherd has a long neck, which is raised when excited and lowered when moving at a fast pace as well as stalking. The tail is bushy and reaches to the hock.'
    
    """set up nlp"""
    nlp = spacy.load("en_core_web_sm")

    get_top_k_attribute_spans_with_whole_image(image, text_raw, class_name, nlp, k = 5)
    
def main_use_patches(output_dir, method):

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    """specify class"""
    class_name = 'Siberian Husky'
    # class_name = 'Persian cat'
    # class_name = 'German_Shepherd'
   
    """load original text"""
    text_raw = 'The Siberian Husky is a medium-sized working sled dog breed. The breed belongs to the Spitz genetic family. It is recognizable by its thickly furred double coat, erect triangular ears, and distinctive markings, and is smaller than the similar-looking Alaskan Malamute.'
    # text_raw = 'The Persian cat is a long-haired breed of cat characterized by its round face and short muzzle. It is also known as the \"Persian Longhair\" in the English-speaking countries. The first documented ancestors of the Persian were imported into Italy from Persia around 1620. Recognized by the cat fancy since the late 19th century, it was developed first by the English, and then mainly by American breeders after the Second World War. Some cat fancier organizations\' breed standards subsume the Himalayan and Exotic Shorthair as variants of this breed, while others treat them as separate breeds.'
    # text_raw = 'They have a domed forehead, a long square-cut muzzle with strong jaws and a black nose. The eyes are medium-sized and brown. The ears are large and stand erect, open at the front and parallel, but they often are pulled back during movement. A German Shepherd has a long neck, which is raised when excited and lowered when moving at a fast pace as well as stalking. The tail is bushy and reaches to the hock.'
    
    """set up nlp"""
    nlp = spacy.load("en_core_web_sm")

    """get text """
    ## using dependency parer:
    if method == 'DependencyParser':
        if_use_template = False
        noun_chunk_list, prompts = get_attribute_candidate_prompts(text_raw, nlp, if_use_template = if_use_template, class_name = class_name)
        print('prompts:', prompts)
    elif method == 'AMR':
        ## using AMR: TODO:
        noun_chunk_list = ['sled dog', 'dog breed', 'genetic family', 'double coat', 'erect triangular ears', 'distinctive markings']   
        prompts = ['sled dog', 'dog breed', 'genetic family', 'double coat', 'erect triangular ears', 'distinctive markings']   
    
    """load image patches"""
    img_path_list = ['/shared/nas/data/m1/wangz3/multimedia_attribute/AttDiscovery/test_images/Husky_L.jpg']
    inputs, raw_images = get_detectron_inputs(img_path_list)
    
    """get region proposal"""
    image_inputs, cfg = get_region_proposals(inputs, raw_images, k=1000)

    # NOTE: use the first image:
    raw_image = image_inputs[0]['raw_image']
    image_patches = image_inputs[0]['raw_image_patches']
    proposal_boxes = list(image_inputs[0]['proposals'])

    """run clip"""
    probs_per_image, probs_per_text = get_probs(model, preprocess, noun_chunk_list, prompts, image_patches, device)
    print(probs_per_image.shape)
    print(probs_per_text.shape)
    #TODO: for each text prompt, find best matching image patch; then take the most confident k pairs as the extracted attribute-patch pair
    print(probs_per_text)
    best_match_image_patch_idx = np.argmax(probs_per_text, axis=1)
    best_match_image_patch_score = np.max(probs_per_text, axis=1)
    print(best_match_image_patch_idx)
    print(best_match_image_patch_score)
    
    # take highest alpha% text spans
    alpha = 1
    k = int(len(noun_chunk_list)*alpha)
    ranked_text_idx = np.argsort(-1*best_match_image_patch_score)[:k]
    print(ranked_text_idx)

    for rank_idx in range(len(ranked_text_idx)):
        text_idx = ranked_text_idx[rank_idx]
        text_span = noun_chunk_list[text_idx]
        image_patch_idx = best_match_image_patch_idx[text_idx]
        box_coord = proposal_boxes[image_patch_idx].to("cpu")
        print(text_idx,'|',text_span,'|',box_coord)

        v = Visualizer(raw_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) # bgr->rgb
        out = v.draw_box(box_coord, alpha=0.8, edge_color='g', line_style='-')
        out = v.draw_text(f'{text_span}',(box_coord[0],box_coord[1])) 
        cv2.imwrite(os.path.join(output_dir, f'rank-{rank_idx}_textidx-{text_idx}.png'),out.get_image()[:, :, ::-1])

if __name__ == '__main__':
    method = 'AMR' # DependencyParser
    method = 'DependencyParser' 
    output_dir = f'/shared/nas/data/m1/wangz3/multimedia_attribute/AttDiscovery/test_output/text_patch_pairs/{method}'
    main_use_patches(output_dir, method)




    




