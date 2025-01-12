from rapidfuzz import fuzz
import pandas as pd
import time
from tqdm import tqdm

def trigram_similarity(a, b):
    return fuzz.ratio(a, b) / 100

def find_similar_products(dfs, column, threshold=0.8, max_results="all"):
    similar_products = []
    start_time = time.time()

    df_names = {
        "Megastroy": dfs[0],
        "Saturn": dfs[1],
        "Obi": dfs[2]
    }

    total_products = len(dfs[0])
    print(f"\nОбработка датафреймов")
    
    test_sample = min(250, total_products)
    test_start = time.time()
    for i in range(test_sample):
        product1 = dfs[0][column].iloc[i]
        _ = [p for p in dfs[1][column] if trigram_similarity(product1.lower(), p.lower()) >= threshold]
        _ = [p for p in dfs[2][column] if trigram_similarity(product1.lower(), p.lower()) >= threshold]
    test_time = time.time() - test_start
    
    estimated_time = (test_time / test_sample) * total_products
    if estimated_time / 60 > 60:
        print(f"Ориентировочное время выполнения: {estimated_time / 3600:.1f} часов")
    else:
        print(f"Ориентировочное время выполнения: {estimated_time / 60:.1f} минут")
    print("-" * 80)

    for product1 in tqdm(dfs[0][column], 
                        total=total_products,
                        desc="Поиск похожих товаров",
                        colour="green",
                        ncols=100):
        similar_in_df2 = [
            product2 for product2 in dfs[1][column]
            if trigram_similarity(product1.lower(), product2.lower()) >= threshold
        ]
        similar_in_df3 = [
            product3 for product3 in dfs[2][column]
            if trigram_similarity(product1.lower(), product3.lower()) >= threshold
        ]

        if similar_in_df2 and similar_in_df3:
            similar_products.append({
                'DF1 (Megastroy)': product1,
                'DF2 (Saturn)': similar_in_df2[0],
                'DF3 (Obi)': similar_in_df3[0]
            })

        if isinstance(max_results, int) and len(similar_products) >= max_results:
            break

    end_time = time.time()
    print(f"\nОбщее время обработки: {end_time - start_time:.2f} секунд")

    return similar_products

def save_to_excel(results, filename):
    if results:
        df = pd.DataFrame(results)
        df.to_excel(filename, index=False)
        print(f"Результаты сохранены в файл {filename}")
    else:
        print("Нет данных для сохранения.")

try:
    df1 = pd.read_csv('megastroy.csv')
    df2 = pd.read_csv('saturn.csv')
    df3 = pd.read_csv('obi.csv')
except FileNotFoundError as e:
    print(f"Ошибка: {e}")
    exit()

column = 'Product Name'
threshold = 0.8
max_results = "all"

print("\n\nНачало поиска похожих товаров")

results = find_similar_products([df1, df2, df3], column, threshold, max_results)

if results:
    print("Найдены похожие товары:")
    save_to_excel(results, 'similfar_products.xlsx')
else:
    print("Похожие товары не найдены.")
