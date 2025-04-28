"""
"""
import os
import re
import json
import random
from collections import defaultdict
from ast import literal_eval
from decimal import Decimal

import cleantext
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from flask import render_template_string
from rich import print
from pyserini.search.lucene import LuceneSearcher

from web_agent_site.utils import (
    BASE_DIR,
    DEFAULT_FILE_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs']
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
    keywords,
    search_engine,
    all_products,
    product_item_dict,
    attribute_to_asins=None,
    use_llm=False,  # Default to False to maintain backward compatibility
):
    # Special search modes remain unchanged
    if isinstance(keywords, list) and len(keywords) > 0 and keywords[0] in ['<r>', '<a>', '<c>', '<q>']:
        if keywords[0] == '<r>':
            top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
        elif keywords[0] == '<a>':
            attribute = ' '.join(keywords[1:]).strip()
            asins = attribute_to_asins[attribute]
            top_n_products = [p for p in all_products if p['asin'] in asins]
        elif keywords[0] == '<c>':
            category = keywords[1].strip()
            top_n_products = [p for p in all_products if p['category'] == category]
        elif keywords[0] == '<q>':
            query = ' '.join(keywords[1:]).strip()
            top_n_products = [p for p in all_products if p['query'] == query]
    else:
        # Standard keyword search path
        if isinstance(keywords, list):
            keywords_str = ' '.join(keywords)
        else:
            keywords_str = keywords  # Handle the case where keywords is already a string
            keywords = keywords_str.split()  # Convert to list for LLM ranking

        # Initial BM25 search with Lucene
        hits = search_engine.search(keywords_str, k=SEARCH_RETURN_N * (3 if use_llm else 1))
        
        if use_llm:
            # Store more information for LLM ranking
            docs = [search_engine.doc(hit.docid) for hit in hits]
            top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
            
            # Create candidate products with scores
            candidate_products = []
            for i, asin in enumerate(top_n_asins):
                if asin in product_item_dict:
                    product = product_item_dict[asin].copy()  # Make a copy to avoid modifying original
                    product['initial_score'] = hits[i].score
                    candidate_products.append(product)
            
            # Apply LLM-based ranking
            try:
                # Get Ollama Mistral ranking
                ordered_indices = rank_with_ollama_mistral_cli(candidate_products, keywords_str)
                
                if ordered_indices and len(ordered_indices) > 0:
                    # Apply LLM ranking
                    num_ranked = min(8, len(candidate_products))
                    valid_indices = [idx for idx in ordered_indices if 0 <= idx < num_ranked]
                    
                    # Create final product list
                    llm_ranked_products = [candidate_products[:num_ranked][idx] for idx in valid_indices]
                    remaining_products = [p for i, p in enumerate(candidate_products[:num_ranked]) 
                                        if i not in valid_indices]
                    remaining_products.extend(candidate_products[num_ranked:])
                    
                    # Sort remaining by original score
                    remaining_products.sort(key=lambda x: x.get('initial_score', 0), reverse=True)
                    
                    # Create final list
                    ranked_candidate_products = llm_ranked_products + remaining_products
                    
                    # Extract just the product objects for the final result
                    top_n_products = ranked_candidate_products[:SEARCH_RETURN_N]
                else:
                    # Fall back to traditional approach
                    top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
            except Exception as e:
                print(f"LLM ranking failed: {e}")
                # Fall back to traditional approach
                top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
        else:
            # Original approach - direct conversion from hits to products
            docs = [search_engine.doc(hit.docid) for hit in hits]
            top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
            top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    
    return top_n_products

def rank_with_ollama_mistral_cli(candidate_products, keywords_str, num_to_rank=8):
    """Rank products using Ollama CLI"""
    import subprocess
    import re
    
    # Create prompt
    llm_prompt = f"Query: {keywords_str}\n\nRank these products by relevance to the query (most relevant first):\n\n"
    
    # Create product summaries
    product_summaries = []
    for i, product in enumerate(candidate_products[:num_to_rank]):
        summary = f"Product {i+1}:\n"
        summary += f"Title: {product.get('Title', 'N/A')}\n"
        summary += f"Category: {product.get('category', 'N/A')}\n"
        
        # Add limited attributes - handle both dict and list formats safely
        if 'Attributes' in product:
            attrs = product['Attributes']
            summary += "Key Attributes:\n"
            
            # Handle different attribute formats
            if isinstance(attrs, dict):
                # It's a dictionary, take first few items
                for k, v in list(attrs.items())[:3]:
                    summary += f"- {k}: {v}\n"
            elif isinstance(attrs, list):
                # It's a list, take first few items
                for attr in attrs[:3]:
                    if isinstance(attr, dict):
                        for k, v in attr.items():
                            summary += f"- {k}: {v}\n"
                    else:
                        summary += f"- {attr}\n"
            else:
                # It's something else (string, number, etc.)
                summary += f"- {attrs}\n"
        
        product_summaries.append(summary)
    
    llm_prompt += "\n".join(product_summaries)
    llm_prompt += "\n\nReturn only a comma-separated list of product numbers, from most to least relevant."
    
    try:
        # Call ollama through CLI and provide input directly
        print("Running Ollama Mistral for product ranking...")
        result = subprocess.run(
            ["ollama", "run", "mistral:latest"],
            input=llm_prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            llm_response = result.stdout
            print(f"Ollama response: {llm_response}")
            # Parse response to get ordered indices
            matches = re.search(r'(\d+(?:,\s*\d+)*)', llm_response)
            if matches:
                ordered_indices = [int(idx.strip()) - 1 for idx in matches.group(1).split(',')]
                print(f"Parsed ordered indices: {ordered_indices}")
                return ordered_indices
            else:
                print(f"No indices found in response")
        else:
            print(f"Ollama CLI error: {result.stderr}")
    except Exception as e:
        print(f"Ollama CLI call failed: {e}")

# def get_top_n_product_from_keywords(
#         keywords,
#         search_engine,
#         all_products,
#         product_item_dict,
#         attribute_to_asins=None,
#     ):
#     if keywords[0] == '<r>':
#         top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
#     elif keywords[0] == '<a>':
#         attribute = ' '.join(keywords[1:]).strip()
#         asins = attribute_to_asins[attribute]
#         top_n_products = [p for p in all_products if p['asin'] in asins]
#     elif keywords[0] == '<c>':
#         category = keywords[1].strip()
#         top_n_products = [p for p in all_products if p['category'] == category]
#     elif keywords[0] == '<q>':
#         query = ' '.join(keywords[1:]).strip()
#         top_n_products = [p for p in all_products if p['query'] == query]
#     else:
#         keywords = ' '.join(keywords)
#         hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
#         docs = [search_engine.doc(hit.docid) for hit in hits]
#         top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
#         top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
#     return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'../search_engine/{indexes}'))
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def load_products(filepath, num_products=None, human_goals=True):
    # TODO: move to preprocessing step -> enforce single source of truth
    with open(filepath) as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
    
    # with open(DEFAULT_REVIEW_PATH) as f:
    #     reviews = json.load(f)
    all_reviews = dict()
    all_ratings = dict()
    # for r in reviews:
    #     all_reviews[r['asin']] = r['reviews']
    #     all_ratings[r['asin']] = r['average_rating']

    if human_goals:
        with open(HUMAN_ATTR_PATH) as f:
            human_attributes = json.load(f)
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes = json.load(f)
    print('Attributes loaded.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        products = products[:num_products]
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10:
            continue

        if asin in asins:
            continue
        else:
            asins.add(asin)

        products[i]['category'] = p['category']
        products[i]['query'] = p['query']
        products[i]['product_category'] = p['product_category']

        products[i]['Title'] = p['name']
        products[i]['Description'] = p['full_description']
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p['small_description'] \
            if isinstance(p['small_description'], list) else [p['small_description']]

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            pricing = [
                float(Decimal(re.sub(r'[^\d.]', '', price)))
                for price in pricing.split('$')[1:]
            ]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p['customization_options']
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image

        # without color, size, price, availability
        # if asin in attributes and 'attributes' in attributes[asin]:
        #     products[i]['Attributes'] = attributes[asin]['attributes']
        # else:
        #     products[i]['Attributes'] = ['DUMMY_ATTR']
        # products[i]['instruction_text'] = \
        #     attributes[asin].get('instruction', None)
        # products[i]['instruction_attributes'] = \
        #     attributes[asin].get('instruction_attributes', None)

        # without color, size, price, availability
        if asin in attributes and 'attributes' in attributes[asin]:
            products[i]['Attributes'] = attributes[asin]['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        if human_goals:
            if asin in human_attributes:
                products[i]['instructions'] = human_attributes[asin]
        else:
            products[i]['instruction_text'] = \
                attributes[asin].get('instruction', None)

            products[i]['instruction_attributes'] = \
                attributes[asin].get('instruction_attributes', None)

        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()

        all_products.append(products[i])

    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins
