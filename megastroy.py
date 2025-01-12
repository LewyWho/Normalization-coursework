import requests
from bs4 import BeautifulSoup
import csv

# URL главной страницы каталога
url = 'https://megastroy.com/catalog'
base_url = 'https://megastroy.com'

def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Ошибка при запросе URL {url}: {e}")
        return None

# Функция для парсинга страницы товаров с проверкой на пагинацию
def parse_products_page(page_url, category_name, subcategory_name, subcategory_href, second_level_subcategory_name, second_level_subcategory_href, writer):
    while page_url:
        product_response = requests.get(page_url)
        
        if product_response.status_code == 200:
            product_soup = BeautifulSoup(product_response.content, 'html.parser')

            # Извлекаем товары на странице
            products = product_soup.find_all('div', class_='products-list__content-title')

            if products:
                for product in products:
                    product_name = product.find('a').get('title').strip()
                    print(f'--- {product_name}')
                    writer.writerow([category_name, subcategory_name, subcategory_href, second_level_subcategory_name, second_level_subcategory_href, product_name])
            else:
                print(f'На странице {page_url} не найдено товаров.')
                writer.writerow([category_name, subcategory_name, subcategory_href, second_level_subcategory_name, second_level_subcategory_href, "No products found"])

            # Проверяем, есть ли следующая страница
            next_page = product_soup.find('a', class_='pagination__item pagination__arrow', href=True)
            if next_page:
                page_url = base_url + next_page['href']
                print(f'Переход на следующую страницу: {page_url}')
            else:
                print(f'Нет следующей страницы для {page_url}.')
                page_url = None
        else:
            print(f"Ошибка при запросе страницы товаров {page_url}, статус код: {product_response.status_code}")
            page_url = None

# Основная логика парсинга каталога
def parse_catalog():
    html = get_html(url)
    
    if not html:
        print("Не удалось загрузить главную страницу каталога.")
        return
    
    soup = BeautifulSoup(html, 'html.parser')

    categories = soup.find_all('div', class_='catalog-list__item-four')

    if not categories:
        print("Категории не найдены на главной странице.")
        return

    csv_file_path = "megastroy.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Category Name", "Subcategory Name", "Subcategory URL", "Second-level Subcategory Name", "Second-level Subcategory URL", "Product Name"])

        # Проходим по каждой категории
        for category in categories:
            try:
                category_name = category.find('a', class_='catalog-list__item-title').text.strip()
                category_href = base_url + category.find('a', class_='catalog-list__item-title')['href']
            except AttributeError:
                print("Ошибка извлечения названия или ссылки категории.")
                continue

            print(f'{category_name} - {category_href}')
            
            sub_response = requests.get(category_href)
            if sub_response.status_code != 200:
                print(f"Ошибка при запросе подкатегорий для {category_name}, статус код: {sub_response.status_code}")
                continue

            sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
            subcategories = sub_soup.find_all('a', class_='categories-list__content-title')

            if not subcategories:
                print(f"Подкатегории не найдены для категории {category_name}.")
                writer.writerow([category_name, "No subcategory", category_href, "", "", ""])
                continue

            for subcategory in subcategories:
                try:
                    subcategory_name = subcategory.text.strip()
                    subcategory_href = base_url + subcategory['href']
                except AttributeError:
                    print("Ошибка извлечения подкатегории.")
                    continue

                print(f'- {subcategory_name} - {subcategory_href}')

                second_sub_response = requests.get(subcategory_href)
                if second_sub_response.status_code != 200:
                    print(f"Ошибка при запросе товаров для {subcategory_name}, статус код: {second_sub_response.status_code}")
                    continue

                second_sub_soup = BeautifulSoup(second_sub_response.content, 'html.parser')
                second_level_subcategories = second_sub_soup.find_all('a', class_='categories-list__content-title')

                if not second_level_subcategories:
                    print(f"В подкатегории {subcategory_name} не найдено подкатегорий второго уровня.")
                    writer.writerow([category_name, subcategory_name, subcategory_href, "No second-level subcategory", "", ""])
                    continue

                for second_level_subcategory in second_level_subcategories:
                    try:
                        second_level_subcategory_name = second_level_subcategory.text.strip()
                        second_level_subcategory_href = base_url + second_level_subcategory['href']
                    except AttributeError:
                        print("Ошибка извлечения подкатегории второго уровня.")
                        continue

                    print(f'-- {second_level_subcategory_name} - {second_level_subcategory_href}')

                    # Переход на первую страницу товаров и парсинг с учётом пагинации
                    parse_products_page(second_level_subcategory_href, category_name, subcategory_name, subcategory_href, second_level_subcategory_name, second_level_subcategory_href, writer)

if __name__ == "__main__":
    parse_catalog()
    print("Парсинг завершён")
