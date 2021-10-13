'''
1. Important keys of each product:
    - `bullet_point`
        - Content: Important features of the products
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `color`
        - Content: Color of the product as text
        - Format: `[{"language_tag": <str>, "standardized_values": [<str>],
            "value": <str>}, ...]`
    - `country`
        - Content: Country of the marketplace, as an
            [ISO 3166-1 alpha 2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
            code
        - Format: `<str>`
    - `domain_name`
        - Content: Domain name of the marketplace where the product is found.
            A product listing in this collection is uniquely identified by
            (`item_id`, `domain_name`)
        - Format: `<str>`
    - `item_dimensions`
        - Content: Dimensions of the product (height, width, length)
        - Format: `{"height": {"normalized_value": {"unit": <str>, "value":
            <float>}, "unit": <str>, "value": <float>}, "length":
            {"normalized_value": {"unit": <str>, "value": <float>}, "unit": <str>,
            "value": <float>}, "width": {"normalized_value": {"unit": <str>,
            "value": <float>}, "unit": <str>, "value": <float>}}}`
    - `item_id`
        - Content: The product reference id. A product listing in this
            collection is uniquely identified by (`item_id`, `domain_name`).
            A corresponding product page may exist at
            `https://www.<domain_name>/dp/<item_id>`
        - Format: `<str>`
    - `item_keywords`
        - Content: Keywords for the product
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `item_name`
        - Content: The product name
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `item_shape`
        - Content: Description of the product shape
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `item_weight`
        - Content: The product weight
        - Format: `[{"normalized_value": {"unit": <str>, "value": <float>},
            "unit": <str>, "value": <float>}, ...]`
    - `main_image_id`
        - Content: The main product image, provided as an `image_id`. See the
            descripton of `images/metadata/images.csv.gz` below
        - Format: `<str>`
    - `material`
        - Content: Description of the product material
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `model_name`
        - Content: Model name
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `pattern`
        - Content: Product pattern
        - Format: `[{ "language_tag": <str>, "value": <int> }, ...]`
    - `product_description`
        - Content: Product description as HTML 
        - Format: `[{ "language_tag": <str>, "value": <int> }, ...]`
    - `product_type`
        - Content: Product type (category)
        - Format: `<str>`
    - `style`
        - Content: Style of the product
        - Format: `[{ "language_tag": <str>, "value": <int> }, ...]`
    
2. A datasample of a product:
    {"item_dimensions": 
        {"height": {"normalized_value": {"unit": "inches", "value": 12}, "unit": "inches", "value": 12}, 
        "length": {"normalized_value": {"unit": "inches", "value": 12}, "unit": "inches", "value": 12}, 
        "width": {"normalized_value": {"unit": "inches", "value": 1.5}, "unit": "inches", "value": 1.5}}, 
    "bullet_point": [
        {"language_tag": "en_US", "value": "These vintage lawn chairs may have seen better days, but they have obviously had a rebirth. Brightly painted, they've been repurposed as the hub of hang-out spot outside a warehouse. This colorful urban-look piece will add a bright spot to any room."}, 
        {"language_tag": "en_US", "value": "A modern colorful print with a vintage twist"}, 
        {"language_tag": "en_US", "value": "Printed on wood, framed in a white-painted wood frame"}, 
        {"language_tag": "en_US", "value": "12\" x 12\""}, 
        {"language_tag": "en_US", "value": "Made to order"}], 
    "color": [
        {"language_tag": "en_US", "standardized_values": ["Multi"], "value": "Multicolor"}], 
    "item_id": "B073P5PZ5P", 
    "item_name": [
        {"language_tag": "zh_CN", "value": "Rivet \u590d\u53e4\u84dd\u8272\u9ec4\u8272\u548c\u7eff\u8272\u6905\u5b50 \u9ed1\u8272\u6728\u6846\u5899\u58c1\u827a\u672f"}, 
        {"language_tag": "en_US", "value": "Amazon Brand \u2013 Rivet Vintage Blue Yellow and Green Chairs in White Wood Frame Wall Art, 12\" x 12\""}], 
    "item_weight": [{"normalized_value": {"unit": "pounds", "value": 2.5}, "unit": "pounds", "value": 2.5}], 
    "model_number": [{"value": "16523-frwa30"}], 
    "product_type": [{"value": "HOME"}], 
    "style": [{"language_tag": "en_US", "value": "White"}], 
    "main_image_id": "91e1hw35cDL", 
    "item_keywords": [
        {"language_tag": "en_US", "value": "framed-prints"}, 
        {"language_tag": "en_US", "value": "wall art"}, 
        {"language_tag": "en_US", "value": "wall decor"}, 
        {"language_tag": "en_US", "value": "canvas wall art"}, 
        {"language_tag": "en_US", "value": "wall art for living room"}, 
        {"language_tag": "en_US", "value": "bathroom decor"}, 
        {"language_tag": "en_US", "value": "posters"}, 
        {"language_tag": "en_US", "value": "framed wall art"}, 
        {"language_tag": "en_US", "value": "wall decorations for living room"}, 
        {"language_tag": "en_US", "value": "living room decor"}, 
        {"language_tag": "en_US", "value": "cuadros de pared de sala"}, 
        {"language_tag": "en_US", "value": "Rivet"}, 
        {"language_tag": "en_US", "value": "mid century"}, 
        {"language_tag": "en_US", "value": "modern"}, 
        {"language_tag": "en_US", "value": "Multi"}, 
        {"language_tag": "en_US", "value": "Multi"}, 
        {"language_tag": "en_US", "value": "12\"x12\""}], 
    "country": "US", 
    "domain_name": "amazon.com"
'''
import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import glob


data_root = '/media/hxd/82231ee6-d2b3-4b78-b3b4-69033720d8a8/MyDatasets/amazon'
attr = 'color' # product_type
max_num_values_per_attr = 25

# load product
products = []
json_files = glob.glob(data_root + '/metadata/*.json')
for json_file in json_files:
    for line in open(json_file, 'r'):
        products.append(json.loads(line))
# load image
img_info = pd.read_csv(data_root + '/images.csv')

flag_auto_attr_value = True
if not flag_auto_attr_value:
    ## (1) manually define attribute values
    attr_list = {
        'color':['Black', 'White', 'Blue', 'Brown', 'Gray', 'Orange', 'Red', 'Yellow', 'Pink', 'Silver', 'Bronze', 'Cream', 'Walnut'], 
        'material':['Leather', 'Metal', 'Plastic', 'Glass', 'Rubber', 'Stoneware', 'Wood', 'Fabric', 'Memory_foam'],
        'item_shape':['Rectangular', 'Ellipsoidal', 'Cubic', 'Round', 'Long', 'L-Shape'],
        'style':['Modern', 'Contemporary', 'Traditional', 'Classic']}
    att_values = attr_list[attr]
    att_values = [x.lower() for x in att_values]
else:
    ## (2) obtain attribute values by frequency
    tmp_dic = {}
    for product in products:
        if attr in product.keys():
            if attr == 'product_type':
                att_value = [x['value'] for x in product[attr]]
            else:
                att_value = [x['value'] for x in product[attr] if x['language_tag'] == 'en_US']
            if len(att_value) > 0:
                att_value = att_value[0].lower()
                if att_value in tmp_dic.keys():
                    tmp_dic[att_value] += 1
                else:
                    tmp_dic[att_value] = 1

    top_values = dict(sorted(tmp_dic.items(), key=lambda item: item[1], reverse=True))
    att_values = list(top_values.keys())#[:max_num_values_per_attr]

for value in att_values:
    if not os.path.exists(data_root + '/img_by_attr/' + attr + '/' + value):
        os.makedirs(data_root + '/img_by_attr/' + attr + '/' + value)


product_descr = {}
for product in products:
    description = []
    # if product['country'] == 'US' and \
    if attr in product.keys() and \
       'main_image_id' in product.keys():
        if attr == 'product_type':
            att_value = [x['value'] for x in product[attr]]
        else:
            att_value = [x['value'] for x in product[attr] if x['language_tag'] == 'en_US']
        if len(att_value) > 0:
            att_value = att_value[0].lower()
            if att_value in att_values:
                img_id = product['main_image_id']
                img_path = img_info.loc[img_info['image_id'] == img_id]['path'].values[0]
                img = cv2.imread(data_root + '/small/' + img_path)
                cv2.imwrite(data_root + '/img_by_attr/' + attr + '/' + att_value + '/' + img_id + '.jpg', img)
                if 'bullet_point' in product.keys():
                    description = [x['value'] for x in product['bullet_point'] if x['language_tag'] == 'en_US']
                    product_descr[img_id] = description

with open(data_root + '/img_by_attr/' + attr + '/product_description.json', 'w') as json_file:
    json.dump(product_descr, json_file)