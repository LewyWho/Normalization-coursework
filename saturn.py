import requests
from bs4 import BeautifulSoup
import csv

base_url = "https://nnv.saturn.net/catalog"
main_url = 'https://nnv.saturn.net'

def get_html(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Функция для парсинга продуктов на странице
def parse_products_on_page(soup, writer, category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href):
    products = soup.find_all('li', class_='catalog_Level2__goods_list__item')
    
    if products:
        for product in products:
            product_name = product.find('a', class_='goods_card_text swiper-no-swiping').text.strip()
            print(f'--- {product_name}')
            writer.writerow([category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href, product_name])  # Запись продукта
    else:
        print(f'--- No products found in {second_subcategory_name}')
        writer.writerow([category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href, "No products found"])

# Функция для получения всех страниц с продуктами
def parse_pagination(soup, base_href):
    pagination = soup.select('.pagination__item a.pagination__link')
    page_links = []

    if pagination:
        for page in pagination:
            page_number = page.get('data-page')
            if page_number:
                page_href = base_href + "?page=" + page_number
                page_links.append(page_href)
    
    return page_links

# Функция для парсинга главной категории
def parse_main_category():
    html = get_html(base_url)
    soup = BeautifulSoup(html, 'html.parser')
    categories = soup.select('.catalog__level1__nav-list__item')

    # Создание CSV-файла
    with open('saturn.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Main Category", "Subcategory", "Subcategory URL", "Second Level Subcategory Name", "Second Level Subcategory URL", "Product Name"])  # Заголовки

        for category in categories:
            category_name = category.find('span', class_='catalog__level1__nav-list__item__title').text.strip()
            category_href = main_url + category.find('a')['href']

            print(f'{category_name} - {category_href}')
            
            sub_response = requests.get(category_href)

            if sub_response.status_code == 200:
                sub_soup = BeautifulSoup(sub_response.content, 'html.parser')

                subcategories = sub_soup.find_all('a', class_='catalog__level2__nav-list__item')

                if subcategories:
                    for subcategory in subcategories:
                        subcategory_name = subcategory.find('span', class_='catalog__level2__nav-list__item__title').text.strip()
                        subcategory_href = main_url + subcategory['href']

                        print(f'- {subcategory_name} - {subcategory_href}')
                        writer.writerow([category_name, subcategory_name, subcategory_href, "", "", ""])  # Запись подкатегории

                        second_sub_response = requests.get(subcategory_href)

                        if second_sub_response.status_code == 200:
                            second_sub_soup = BeautifulSoup(second_sub_response.content, 'html.parser')

                            second_level_subcategories = second_sub_soup.find_all('a', class_='catalog__level2__nav-list__item')

                            if second_level_subcategories:
                                for second_subcategory in second_level_subcategories:
                                    second_subcategory_name = second_subcategory.find('span', class_='catalog__level2__nav-list__item__title').text.strip()
                                    second_subcategory_href = main_url + second_subcategory['href']

                                    print(f'-- {second_subcategory_name} - {second_subcategory_href}')
                                    writer.writerow([category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href, ""])  # Запись второго уровня подкатегории

                                    product_response = requests.get(second_subcategory_href)

                                    if product_response.status_code == 200:
                                        product_soup = BeautifulSoup(product_response.content, 'html.parser')
                                        
                                        parse_products_on_page(product_soup, writer, category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href)

                                        pagination_links = parse_pagination(product_soup, second_subcategory_href)

                                        if pagination_links:
                                            for page_link in pagination_links:
                                                page_response = requests.get(page_link)

                                                if page_response.status_code == 200:
                                                    page_soup = BeautifulSoup(page_response.content, 'html.parser')
                                                    parse_products_on_page(page_soup, writer, category_name, subcategory_name, subcategory_href, second_subcategory_name, second_subcategory_href)
                                                else:
                                                    print(f"Ошибка при запросе страницы {page_link}, статус код: {page_response.status_code}")
                                        else:
                                            print(f'Нет дополнительных страниц для {second_subcategory_name}')
                                    else:
                                        print(f"Ошибка при запросе товаров для {second_subcategory_name}, статус код: {product_response.status_code}")
                            else:
                                print(f'-- No second-level subcategories in {subcategory_name}')
                                writer.writerow([category_name, subcategory_name, subcategory_href, "No second-level subcategory", "", ""])
                        else:
                            print(f"Ошибка при запросе подкатегорий для {subcategory_name}, статус код: {second_sub_response.status_code}")
                else:
                    print(f'- No subcategories found in {category_name}')
                    writer.writerow([category_name, "No subcategory", category_href, "", "", ""])
            else:
                print(f"Ошибка при запросе подкатегорий для {category_name}, статус код: {sub_response.status_code}")

if __name__ == "__main__":
    parse_main_category()
    print("Парсинг завершён")
