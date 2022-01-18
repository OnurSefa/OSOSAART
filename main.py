import requests
import html
import re
import urllib.request
import time
import os

if __name__ == '__main__':
    first_page_response = requests.get('https://www.deviantart.com/search?q=synthwave%20art')
    html_result = html.unescape(first_page_response.text)
    folder = 'synthwave'
    os.system('mkdir {}'.format(folder))
    directory = './{}/'.format(folder)

    p = re.compile('<a data-hook="deviation_link".*?href="(.*?)"')
    n = re.compile('<a class="_3YB38" href="(.*?)"')
    l = re.compile('<img.*?class="_3X6pY".*?src="(.*?)"')
    base = "https://www.deviantart.com"

    max_pages = 40
    for i in range(max_pages):
        print(i)
        result = p.findall(html_result)
        for r in result:
            deviation_response = requests.get(r)
            deviation_result = html.unescape(deviation_response.text)
            l_result = l.findall(deviation_result)
            for l_instance in l_result:
                name = directory+l_instance.split('?')[0].split('/')[-1]
                urllib.request.urlretrieve(l_instance, name)
        n_result = n.findall(html_result)
        next_page = base+n_result[0]
        next_page_response = requests.get(next_page)
        html_result = html.unescape(next_page_response.text)
        time.sleep(60)
    print('a')
