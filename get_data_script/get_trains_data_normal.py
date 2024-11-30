import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm


def download_image(img_url, folder='images'):
    """下载单个图片"""
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        img_name = img_url.split('/')[-1]
        img_path = os.path.join(folder, img_name)
        with open(img_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f'图片已下载: {img_name}')
    except requests.RequestException as e:
        print(f'下载图片失败: {img_url}, 错误: {e}')


def create_folder(folder):
    """创建文件夹"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def main(url, headers):
    """主函数"""
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')
        create_folder('images')

        # 使用tqdm显示进度条
        for img in tqdm(img_tags, desc='下载图片'):
            img_url = img.get('data-src')
            if img_url and img_url.endswith('.webp'):
                img_url = urljoin(url, img_url)
                download_image(img_url)
    else:
        print(f"请求失败，状态码：{response.status_code}")


if __name__ == '__main__':
    url = 'https://www.vcg.com/creative-image/putaoguoshi/?sort=fresh&page=3'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    main(url, headers)