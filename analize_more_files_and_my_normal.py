import pandas as pd

input_file = 'more_files.csv'
output_file = 'normalized_more_files.csv'

df = pd.read_csv(input_file)

normalized_names = []

for idx, name in enumerate(df['Original_Name']):
    print(f"[{idx}] Текущее название товара: {name}")
    normalized_name = input("Введите нормализованное название товара: ")
    normalized_names.append(normalized_name)

df['Manual_normalization'] = normalized_names

df.to_csv(output_file, index=False)
print(f"Файл успешно сохранён: {output_file}")
