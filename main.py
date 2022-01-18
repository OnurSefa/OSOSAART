import requests
import html
import re
import urllib.request

if __name__ == '__main__':
    first_page_response = requests.get('https://www.deviantart.com/search?q=fantasy%20landscape')
    html_result = html.unescape(first_page_response.text)
    # with open('third.html', 'w') as output_file:
    #     output_file.write(html_result)
    p = re.compile('<a data-hook="deviation_link".*?href="(.*?)"')
    result = p.findall(html_result)
    directory = './images/'
    for r in result:
        deviation_response = requests.get(r)
        deviation_result = html.unescape(deviation_response.text)
        # with open('second.html', 'w', encoding='utf-8') as output_file:
        #     output_file.write(deviation_result)
        l = re.compile('<img.*?class="_3X6pY".*?src="(.*?)"')
        l_result = l.findall(deviation_result)
        for l_instance in l_result:
            name = directory+l_instance.split('?')[0].split('/')[-1]
            urllib.request.urlretrieve(l_instance, name)

    # with open('first_page.html', 'w') as output_file:
    #     output_file.write(html.unescape(first_page_response.text))

    print('a')
