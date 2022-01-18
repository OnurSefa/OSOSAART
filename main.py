import requests
import html
import re
import urllib.request

if __name__ == '__main__':
    first_page_response = requests.get('https://www.deviantart.com/search?q=fantasy%20landscape')
    html_result = html.unescape(first_page_response.text)

    p = re.compile('<img loading="lazy".*? src="(.*?)"')
    result = p.findall(html_result)
    print(result)
    directory = './images/'
    for r in result:
        urllib.request.urlretrieve(r, directory+r.split('/')[-1].split('?')[0])

    # with open('first_page.html', 'w') as output_file:
    #     output_file.write(html.unescape(first_page_response.text))

    print('a')
