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

    text = re.sub(r'\b[A-Za-z]{2,6}[-]?\d{3,8}\b', '', text)
    text = re.sub(r'\b[A-Za-z]{2,6}[-]?\d{2,8}[-]?\d{2,8}\b', '', text)
    text = re.sub(r'\b[A-Za-z]+\d+[A-Za-z]*[-]?\d*\b', '', text)
    text = re.sub(r'\bарт[\.\s]?\d{2,8}\b', '', text)
    text = re.sub(r'["\.,;:!]', '', text)
    text = re.sub(r'\b(арт|артикул|упак|модель|для)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_normalization_models():
    models = {
        'roberta_squad': pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
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
            
        if method == 'roberta_squad':
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
        'roberta_squad',
        'distilbert_sst'
    ]
    
    print("Обработка данных...")
    df = process_data(df, methods)
    
    save_normalized_results(df)

if __name__ == "__main__":
  limit = 50
  main(limit)