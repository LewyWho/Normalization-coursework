import requests
from bs4 import BeautifulSoup
import csv

url = 'https://obi.ru/catalog'
base_url = 'https://obi.ru'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    categories = soup.find_all('a', class_='kn7A0')

    csv_file_path = "categories_with_subcategories_and_products.csv"

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Category Name", "Subcategory Name", "Subcategory URL", "Second-level Subcategory Name", "Second-level Subcategory URL", "Product Name"])  # Заголовки

        for category in categories:
            category_name = category.find('span', class_='_17tb-').text
            category_href = base_url + category['href']
            sub_response = requests.get(category_href)
            if sub_response.status_code == 200:
                sub_soup = BeautifulSoup(sub_response.content, 'html.parser')

                subcategories = sub_soup.find_all('a', class_='kn7A0')

                if subcategories:
                    for subcategory in subcategories:
                        subcategory_name = subcategory.find('span', class_='_17tb-').text
                        subcategory_href = base_url + subcategory['href']
                        writer.writerow([category_name, subcategory_name, subcategory_href, "", "", ""])

                        second_sub_response = requests.get(subcategory_href)
                        if second_sub_response.status_code == 200:
                            second_sub_soup = BeautifulSoup(second_sub_response.content, 'html.parser')

                            second_level_subcategories = second_sub_soup.find_all('a', class_='kn7A0')

                            if second_level_subcategories:
                                for second_subcategory in second_level_subcategories:
                                    second_subcategory_name = second_subcategory.find('span', class_='_17tb-').text
                                    second_subcategory_href = base_url + second_subcategory['href']
                                    writer.writerow([category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href, ""])
                                    product_response = requests.get(second_subcategory_href)
                                    if product_response.status_code == 200:
                                        product_soup = BeautifulSoup(product_response.content, 'html.parser')
                                        products = product_soup.find_all('p', class_='_1UlGi')
                                        if products:
                                            for product in products:
                                                product_name = product.text.strip()
                                                writer.writerow([category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href, product_name])
                                        else:
                                            writer.writerow([category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href, "No products found"])
                                    else:
                                        print(f"Ошибка при запросе товаров для {second_subcategory_name}, статус код: {product_response.status_code}")
                            else:
                                writer.writerow([category_name, subcategory_name, subcategory_href, "No second-level subcategory", "", ""])
                else:
                    writer.writerow([category_name, "No subcategory", category_href, "", "", ""])
            else:
                print(f"Ошибка при запросе подкатегорий для {category_name}, статус код: {sub_response.status_code}")
else:
    print(f"Ошибка при запросе страницы, статус код: {response.status_code}")

print("Complete")
