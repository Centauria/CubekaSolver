# -*- coding: utf-8 -*-
import requests

WEBSITE = 'https://www.visca.com/regexdict/'
header = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'origin': 'https://www.visca.com',
    'referer': 'https://www.visca.com/regexdict/',
    'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36'
}


def search(reg):
    respond = requests.post(WEBSITE, {
        'str': reg,
        'ifun': 'if',
        'ccg': 'all'
    })
    print(respond.content)


if __name__ == '__main__':
    search(r'^.......ture$')
