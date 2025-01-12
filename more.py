import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from fuzzywuzzy import process, fuzz
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
import os
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

nltk.download('punkt')

def load_data(limit='all'):
    megastroy_df = pd.read_csv('megastroy.csv', encoding='utf-8-sig')
    obi_df = pd.read_csv('obi.csv', encoding='utf-8-sig')
    saturn_df = pd.read_csv('saturn.csv', encoding='utf-8-sig')

    columns = ['Category Name', 'Subcategory Name', 'Second-level Subcategory Name', 'Product Name']
    merged_df = pd.concat([megastroy_df[columns], obi_df[columns], saturn_df[columns]], ignore_index=True)

    if limit != 'all':
        merged_df = merged_df.head(int(limit))

    return merged_df

def normalize_text(text):
    text = str(text).lower()

    text = re.sub(r'\b[A-Za-z]{2,6}[-]?\d{3,8}\b', '', text)  # SYQE-012124, SYQE0121261
    text = re.sub(r'\b[A-Za-z]{2,6}[-]?\d{2,8}[-]?\d{2,8}\b', '', text)  # SY-14-01
    text = re.sub(r'\b[A-Za-z]+\d+[A-Za-z]*[-]?\d*\b', '', text)  # Другие форматы артикулов
    
    text = re.sub(r'["\.,;:!]', '', text)
    text = re.sub(r'\b(арт|артикул|упак|модель|для)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_normalization_models():
    models = {
        'ruT5': pipeline(
            "text2text-generation",
            model="ai-forever/ruT5-base",
            max_length=128,
            device=0 if torch.cuda.is_available() else -1,
        ),
        'rugpt': pipeline(
            "text-generation",
            model="ai-forever/rugpt3small_based_on_gpt2",
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=50,
            truncation=True
        ),

        'rubert_tiny2': pipeline(
            "feature-extraction",
            model="cointegrated/rubert-tiny2",
            device=0 if torch.cuda.is_available() else -1
        ),

        'multilingual_e5': SentenceTransformer('intfloat/multilingual-e5-large'),
        'multilingual_use': SentenceTransformer('distiluse-base-multilingual-cased-v1'),
        'labse': SentenceTransformer('sentence-transformers/LaBSE'),
        'xlm_roberta': pipeline(
            "feature-extraction",
            model="xlm-roberta-base",
            device=0 if torch.cuda.is_available() else -1
        ),
        'rembert': pipeline(
            "feature-extraction",
            model="google/rembert",
            device=0 if torch.cuda.is_available() else -1
        ),
        'spacy': spacy.load('ru_core_news_sm'),
        'tfidf': TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            max_features=15000
        ),
        'bert_multilingual': pipeline(
            "feature-extraction",
            model="bert-base-multilingual-cased",
            device=0 if torch.cuda.is_available() else -1,
            ignore_mismatched_sizes=True
        ),
        'distilbert_multilingual': pipeline(
            "feature-extraction",
            model="distilbert-base-multilingual-cased",
            device=0 if torch.cuda.is_available() else -1,
            ignore_mismatched_sizes=True
        ),
        'xlm_roberta_large': pipeline(
            "feature-extraction",
            model="xlm-roberta-large",
            device=0 if torch.cuda.is_available() else -1
        ),
        'mdeberta_v3': pipeline(
            "text-classification",
            model="microsoft/mdeberta-v3-base",
            device=0 if torch.cuda.is_available() else -1
        ),
        'bertopic': SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens'),
        'roberta_squad': pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        ),
        'flan_t5': pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if torch.cuda.is_available() else -1,
            max_length=128
        ),
        'modern_bert': pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
            device=0 if torch.cuda.is_available() else -1
        ),
        'distilbert_sst': pipeline(
            "text-classification",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
    }
    return models

def normalize_with_models(text, models, method, reference_data=None):
    try:
        original_text = text
        text = normalize_text(text)

        def clean_result(result):
            result = normalize_text(result)
            result = re.sub(r'\b[A-Za-z]+-?\d+(?:-?\d+)*\b', '', result)
            return result.strip()
        
        if method in ['rubert', 'rubert_tiny2', 'rubert_conversational', 
                     'ruelectra', 'xlm_roberta', 'rembert', 'bert_multilingual',
                     'distilbert_multilingual', 'xlm_roberta_large', 'tfidf',
                     'multilingual_e5', 'multilingual_use', 'labse']:
            doc = models['spacy'](original_text.lower())
            parts = []
            
            nouns = [token.text for token in doc if token.pos_ == 'NOUN']
            if len(nouns) >= 2:
                parts.append(' '.join(nouns[:2]))
            elif nouns:
                parts.append(nouns[0])
            
            quantities = re.findall(r'\d+\s*(?:шт|компл|набор|уп)', original_text.lower())
            if quantities:
                parts.append(quantities[0])

            sizes = re.findall(r'\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?\s*(?:см|мм|м|мл|л|г|кг)', original_text.lower())
            if sizes:
                parts.append(sizes[0])
            
            colors = ['медь', 'красный', 'синий', 'зеленый', 'золото', 'серебро', 
                     'белый', 'черный', 'желтый', 'коричневый', 'бежевый',
                     'фиолетовый', 'шампанское']
            found_colors = []
            for color in colors:
                if color in original_text.lower():
                    found_colors.append(color)
            if found_colors:
                if len(found_colors) == 1:
                    parts.append(f"цвет - {found_colors[0]}")
                else:
                    parts.append(f"цвет - {'+'.join(found_colors)}")
            
            return ', '.join(parts)
            
        elif method == 'mdeberta_v3' or method == 'bertopic':
            try:
                doc = models['spacy'](original_text.lower())
                parts = []
                
                nouns = [token.text for token in doc if token.pos_ == 'NOUN']
                if len(nouns) >= 2:
                    parts.append(' '.join(nouns[:2]))
                elif nouns:
                    parts.append(nouns[0])
                
                quantities = re.findall(r'\d+\s*(?:шт|компл|набор|уп)', original_text.lower())
                if quantities:
                    parts.append(quantities[0])
                
                sizes = re.findall(r'\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?\s*(?:см|мм|м|мл|л|г|кг)', original_text.lower())
                if sizes:
                    parts.append(sizes[0])
                
                colors = ['медь', 'красный', 'синий', 'зеленый', 'золото', 'серебро', 
                         'белый', 'черный', 'желтый', 'коричневый', 'бежевый',
                         'фиолетовый', 'шампанское']
                found_colors = []
                for color in colors:
                    if color in original_text.lower():
                        found_colors.append(color)
                if found_colors:
                    if len(found_colors) == 1:
                        parts.append(f"цвет - {found_colors[0]}")
                    else:
                        parts.append(f"цвет - {'+'.join(found_colors)}")
                
                result = ', '.join(parts)
                return clean_result(result) if result else text
                
            except Exception as e:
                print(f"Ошибка {method}: {e}")
                return clean_result(text)
            
        elif method in ['ruT5', 'rut5_paraphrase', 'mt5_multilingual']:
            prompts = [
                f"Опиши товар кратко с характеристиками: {text}",
                f"Укажи название, количество, размер и цвет товара: {text}",
                f"Преобразуй в формат: название, характеристики, цвет: {text}"
            ]
            results = []
            for prompt in prompts:
                try:
                    result = models[method](prompt, max_length=128, 
                                         num_return_sequences=1)[0]['generated_text']
                    result = clean_text(result)
                    results.append(result.strip())
                except Exception as e:
                    continue
            
            valid_results = [r for r in results if is_valid_result(r)]
            if valid_results:
                best_result = max(valid_results, key=len)
                return format_product_description(best_result)
            return text
            
        elif method in ['rugpt']:
            prompt = f"Опиши товар с характеристиками: {text}"
            result = models[method](prompt)[0]['generated_text']
            result = clean_text(result)
            return format_product_description(result)
            
        elif method in ['rubert', 'rubert_tiny2', 'rubert_conversational', 
                       'ruelectra', 'xlm_roberta', 'rembert', 'bert_multilingual',
                       'distilbert_multilingual', 'xlm_roberta_large']:
            embeddings = models[method](text)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings[0][0])
            
            if reference_data is not None:
                ref_embeddings_list = []
                for ref_text in reference_data:
                    ref_emb = models[method](ref_text)
                    if isinstance(ref_emb, list):
                        ref_emb = np.array(ref_emb[0][0])
                    ref_embeddings_list.append(ref_emb)
                
                ref_embeddings = np.stack(ref_embeddings_list)
                
                similarities = cosine_similarity([embeddings], ref_embeddings)
                most_similar_idx = similarities.argmax()
                return reference_data[most_similar_idx]
            return text
            
        elif method in ['multilingual_e5', 'multilingual_use', 'labse']:
            embeddings = models[method].encode(text, convert_to_tensor=True)
            if reference_data is not None:
                ref_embeddings = models[method].encode(reference_data, convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(embeddings, ref_embeddings)
                most_similar_idx = similarities.argmax().item()
                return reference_data[most_similar_idx]
            return text
            
        elif method == 'spacy':
            doc = models[method](text)
            return extract_product_info(doc, text)
            
        elif method == 'tfidf':
            if reference_data is not None:
                tfidf_matrix = models[method].fit_transform(reference_data)
                text_tfidf = models[method].transform([text])
                similarities = cosine_similarity(text_tfidf, tfidf_matrix)
                most_similar_idx = similarities.argmax()
                return reference_data[most_similar_idx]
            return text
            
        elif method == 'mdeberta_v3':
            try:
                result = models[method](text)
                parts = []
                
                doc = models['spacy'](original_text.lower())
                nouns = [token.text for token in doc if token.pos_ == 'NOUN']
                if len(nouns) >= 2:
                    parts.append(' '.join(nouns[:2]))
                elif nouns:
                    parts.append(nouns[0])
                
                base_info = extract_product_info(doc, text)
                if base_info:
                    base_parts = [p for p in base_info.split(', ') 
                                if not any(noun in p.lower() for noun in nouns)]
                    parts.extend(base_parts)
                
                return ', '.join(parts) if parts else text
            except Exception as e:
                print(f"Ошибка mdeberta_v3: {e}")
                return text
                
        elif method == 'bertopic':
            try:
                doc = models['spacy'](original_text.lower())
                nouns = [token.text for token in doc if token.pos_ == 'NOUN']
                parts = []
                
                if len(nouns) >= 2:
                    parts.append(' '.join(nouns[:2]))
                elif nouns:
                    parts.append(nouns[0])

                base_info = extract_product_info(doc, text)
                if base_info:

                    base_parts = [p for p in base_info.split(', ') 
                                if not any(noun in p.lower() for noun in nouns)]
                    parts.extend(base_parts)
                
                return ', '.join(parts) if parts else text
            except Exception as e:
                print(f"Ошибка bertopic: {e}")
                return text
            
        elif method == 'roberta_squad':
            try:
                questions = [
                    "Что это за товар?",
                    "Какие характеристики у товара?",
                    "Какой цвет у товара?",
                    "Какой размер у товара?"
                ]
                
                answers = []
                for question in questions:
                    result = models[method](
                        question=question,
                        context=original_text
                    )
                    if result['score'] > 0.1:  # Порог уверенности
                        answers.append(result['answer'])
                
                return ', '.join(filter(None, answers)) if answers else text
                
            except Exception as e:
                print(f"Ошибка roberta_squad: {e}")
                return text
            
        elif method == 'flan_t5':
            try:
                prompts = [
                    f"Extract product name and characteristics: {text}",
                    f"Describe this product briefly: {text}",
                    f"What are the main features of: {text}"
                ]
                results = []
                for prompt in prompts:
                    result = models[method](prompt, max_length=128)[0]['generated_text']
                    if result and len(result.strip()) > 0:
                        results.append(clean_result(result))
                return ', '.join(filter(None, results)) if results else text
            except Exception as e:
                print(f"Ошибка flan_t5: {e}")
                return text
                
        elif method == 'modern_bert':
            try:
                candidate_labels = [
                    "color", "size", "material", "quantity", 
                    "product type", "brand", "features"
                ]
                result = models[method](text, candidate_labels)
                relevant_labels = [
                    label for score, label in 
                    zip(result['scores'], result['labels']) 
                    if score > 0.5
                ]
                return ', '.join(relevant_labels) if relevant_labels else text
            except Exception as e:
                print(f"Ошибка modern_bert: {e}")
                return text
                
        elif method == 'distilbert_sst':
            try:
                doc = models['spacy'](original_text.lower())
                parts = []
                
                nouns = [token.text for token in doc if token.pos_ == 'NOUN']
                if nouns:
                    parts.append(' '.join(nouns[:2]))
                
                result = models[method](text)
                if result['score'] > 0.7:
                    parts.append(result['label'])
                
                return ', '.join(parts) if parts else text
            except Exception as e:
                print(f"Ошибка distilbert_sst: {e}")
                return text

        if 'attribute_classifier' in models:
            attributes = extract_attributes(text, models['attribute_classifier'])
            parts.extend(attributes)
            
    except Exception as e:
        print(f"Ошибка при нормализации методом {method}: {e}")
        return text

def clean_text(text):
    """Очистка текста от мусора"""
    text = re.sub(r'[\.]{2,}', '', text)
    text = re.sub(r'[\(\)"]', '', text)
    text = re.sub(r'[^а-яА-Я0-9\s\-,]', '', text)
    return text.strip()

def is_valid_result(text):
    """Проверка валидности результата"""
    return (len(text.split()) >= 3 
            and not text.endswith('...') 
            and not 'нормализ' in text.lower())

def extract_product_info(doc, original_text):
    """
    Извлечение информации о товаре с помощью NLP
    """
    parts = []
    
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    if nouns:
        parts.append(nouns[0])
    
    quantities = re.findall(r'\d+\s*(?:шт|компл|набор|уп)', original_text.lower())
    if quantities:
        parts.append(quantities[0])
    
    sizes = re.findall(r'\d+(?:[.,]\d+)?\s*(?:см|мм|м|мл|л|г|кг)', original_text.lower())
    if sizes:
        parts.append(sizes[0])
    
    materials = ['дерево', 'пластик', 'металл', 'стекло', 'керамика']
    for material in materials:
        if material in original_text.lower():
            parts.append(f"материал - {material}")
            break
    
    colors = ['медь', 'красный', 'синий', 'зеленый', 'золото', 'серебро', 
              'белый', 'черный', 'желтый', 'коричневый', 'бежевый']
    for color in colors:
        if color in original_text.lower():
            parts.append(f"цвет - {color}")
            break
    
    return ', '.join(parts) if parts else original_text

def format_product_description(text):
    """
    Форматирование описания товара
    """
    parts = []
    
    product_types = ['набор', 'комплект', 'упаковка', 'штука', 'шт']
    for ptype in product_types:
        if ptype in text.lower():
            match = re.search(fr'{ptype}\s*\d+', text.lower())
            if match:
                parts.append(match.group())
                break
    
    sizes = re.findall(r'\d+(?:[.,]\d+)?\s*(?:см|мм|м|мл|л|г|кг)', text.lower())
    if sizes:
        parts.extend(sizes)
    
    materials = ['дерево', 'пластик', 'металл', 'стекло', 'керамика']
    for material in materials:
        if material in text.lower():
            parts.append(f"материал - {material}")
            break
    
    colors = ['медь', 'красный', 'синий', 'зеленый', 'золото', 'серебро',
              'белый', 'черный', 'желтый', 'коричневый', 'бежевый']
    for color in colors:
        if color in text.lower():
            parts.append(f"цвет - {color}")
            break
    
    return ', '.join(parts) if parts else text

def process_data(df, methods=None):
    """
    Обработка данных с использованием выбранных методов нормализации
    """
    if methods is None:
        methods = [
            'ruT5',
            'rugpt',
            'rembert',
            'rubert',
            'bert_multilingual',
            'xlm_roberta',
            'multilingual_e5',
            'spacy',
            'tfidf'
        ]
    
    print("Инициализация моделей...")
    models = get_normalization_models()

    reference_data = df['Product Name'].tolist()
    
    print("\nНормализация названий товаров...")
    for method in methods:
        column_name = f'Normalized_{method}'
        tqdm.pandas(desc=f"Нормализация методом {method}")
        df[column_name] = df['Product Name'].progress_apply(
            lambda x: normalize_with_models(x, models, method, reference_data)
        )
    
    return df

def save_normalized_results(df, output_dir='normalized_results'):
    """
    Сохранение результатов нормализации в один CSV файл
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_df = pd.DataFrame()

    results_df['Original_Name'] = df['Product Name']
    results_df['Category'] = df['Category Name']
    results_df['Subcategory'] = df['Subcategory Name']
    results_df['Second_level_Subcategory'] = df['Second-level Subcategory Name']

    normalized_columns = [col for col in df.columns if col.startswith('Normalized_')]
    for col in normalized_columns:
        method_name = col.replace('Normalized_', '')
        results_df[method_name] = df[col]

    output_file = os.path.join(output_dir, f'normalized_results_{timestamp}.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nРезультаты нормализации сохранены в: {output_file}")
    
    return output_file

def extract_attributes(text, model):
    """
    Извлечение атрибутов с помощью предобученной модели
    """
    try:

        predictions = model(text)

        attributes = []
        for pred in predictions:
            if pred['score'] > 0.7:
                label = pred['label']
                if label.startswith('color_'):
                    attributes.append(f"цвет - {label.replace('color_', '')}")
                elif label.startswith('material_'):
                    attributes.append(f"материал - {label.replace('material_', '')}")
        
        return attributes
    except Exception as e:
        print(f"Ошибка при извлечении атрибутов: {e}")
        return []

def main(limit='all'):
    print("Загрузка данных...")
    df = load_data(limit)
    print(f"Загружено {len(df)} строк.")

    methods = [
        'ruT5',
        'rugpt',
        'rembert',
        'rubert_tiny2',
        'multilingual_e5',
        'multilingual_use',
        'labse',
        'xlm_roberta',
        'rembert',
        'spacy',
        'tfidf',
        'bert_multilingual',
        'distilbert_multilingual',
        'xlm_roberta_large',
        'mdeberta_v3',
        'bertopic',
        'roberta_squad',
        'flan_t5',
        'modern_bert',
        'distilbert_sst'
    ]
    
    print("Обработка данных...")
    df = process_data(df, methods)
    
    save_normalized_results(df)
    
    print("\nПримеры нормализации:")
    sample_df = df.sample(min(5, len(df)))
    for _, row in sample_df.iterrows():
        print(f"\nИсходное название: {row['Product Name']}")
        for method in methods:
            normalized_col = f'Normalized_{method}'
            print(f"{method}: {row[normalized_col]}")
        print("-" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', default='all', 
                       help="Количество обрабатываемых строк ('all' или число)")
    parser.add_argument('--output_dir', default='normalized_results',
                       help="Директория для сохранения результатов")
    args = parser.parse_args()
    
    main(limit=args.limit)