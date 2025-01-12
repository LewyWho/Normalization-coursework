import pandas as pd

df1 = pd.read_csv('megastroy.csv')
df2 = pd.read_csv('saturn.csv')
df3 = pd.read_csv('obi.csv')

columns = ['Category Name']

def analyze_categories(df1, df2, df3, columns):
    cat1 = set(df1[columns].apply(tuple, axis=1))
    cat2 = set(df2[columns].apply(tuple, axis=1))
    cat3 = set(df3[columns].apply(tuple, axis=1))

    common_all = cat1 & cat2 & cat3

    common_1_2 = cat1 & cat2 - common_all

    common_2_3 = cat2 & cat3 - common_all

    common_1_3 = cat1 & cat3 - common_all

    unique_1 = cat1 - (cat2 | cat3)
    unique_2 = cat2 - (cat1 | cat3)
    unique_3 = cat3 - (cat1 | cat2)
    
    return {
        "Общие для всех трех": common_all,
        "Общие для 1 и 2": common_1_2,
        "Общие для 2 и 3": common_2_3,
        "Общие для 1 и 3": common_1_3,
        "Уникальные для 1": unique_1,
        "Уникальные для 2": unique_2,
        "Уникальные для 3": unique_3
    }

category_analysis = analyze_categories(df1, df2, df3, columns)

output_data = []
for key, value in category_analysis.items():
    for item in value:
        output_data.append({
            'Category Analysis': key,
            'Category Name': item[0]
        })

output_df = pd.DataFrame(output_data)

output_path = 'category_analysis_only_first_cat1.xlsx'
output_df.to_excel(output_path, index=False)
