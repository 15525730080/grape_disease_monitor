import json
import traceback
from pathlib import Path
import aiohttp
import aiofiles
import asyncio
import os

import requests


class TrainsData(object):

    def __init__(self):
        """
        获取不同数据集的类
        """
        self.sci_db_url_v2 = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/getTreeList?datasetId=76b39c9c435d4035b5076412c2ddcb61&version=V2"
        self.sci_db_base_down_url = "https://www.scidb.cn/api/sdb-download-service/downloadFileMole?fileId={0}"

    def get_tree_list(self, target_url) -> dict:
        resp = requests.get(target_url)
        resp = resp.text
        assert resp, "获取数据失败"
        resp_dict = json.loads(resp)
        return resp_dict.get("data", {})

    def get_sci_db_img(self, is_save=False):
        resp_dict_2 = self.get_tree_list(self.sci_db_url_v2)
        stack = [resp_dict_2]
        res = {}
        while stack:
            item = stack.pop()
            if item.get("children"):
                stack.extend(item.get("children"))
            if "jpg" in item.get("label").lower():
                img_path: str = item.get("path")
                img_path = img_path.strip()
                img_type = img_path.split("/")[-2]
                if img_type in res:
                    if item.get("id") not in res[img_type]:
                        res[img_type].append(self.sci_db_base_down_url.format(item.get("id")))
                else:
                    res[img_type] = [self.sci_db_base_down_url.format(item.get("id"))]
        trains_data_res, tests_data_res = self.split_data2trains_tests(res)
        print(trains_data_res, tests_data_res)
        if is_save:
            asyncio.run(self.download_main(trains_data_res, Path().parent.joinpath("img").joinpath("trains").resolve()))
            asyncio.run(self.download_main(tests_data_res, Path().parent.joinpath("img").joinpath("tests").resolve()))

    def split_data2trains_tests(self, img_urls: dict[str:list]):
        """
        :param img_urls:
        :return: 0.8 训练集 0.2 测试集
        """
        trains_data_res = dict()
        tests_data_res = dict()
        for k, v in img_urls.items():
            split_num = int(0.8 * len(v))
            trains = v[:split_num]
            tests = v[split_num:]
            trains_data_res[k] = trains
            tests_data_res[k] = tests
        return trains_data_res, tests_data_res

    # 创建文件夹以保存图片
    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    # 下载并保存图片
    async def download_image(self, session, url, folder, image_name):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    async with aiofiles.open(os.path.join(folder, image_name), 'wb+') as f:
                        content = await response.read()
                        await f.write(content)
                        await f.flush()
                await asyncio.sleep(3)

        except:
            traceback.print_exc()

    # 处理每个类别的图片下载
    async def download_images_by_category(self, session, base_path, category, urls):
        folder = os.path.join(base_path, category)
        self.create_folder(folder)
        for url in urls:
            image_name = url.split("=")[-1] + ".jpg"
            task = await self.download_image(session, url, folder, image_name)

    async def download_main(self, image_urls, base_path):
        #     # 示例图片URL和类别
        #     image_urls = {
        #         "cats": [
        #             "https://example.com/cat1.jpg",
        #             "https://example.com/cat2.jpg",
        #             "https://example.com/cat3.jpg"
        #         ],
        #         "dogs": [
        #             "https://example.com/dog1.jpg",
        #             "https://example.com/dog2.jpg",
        #             "https://example.com/dog3.jpg"
        #         ],
        #         "birds": [
        #             "https://example.com/bird1.jpg",
        #             "https://example.com/bird2.jpg",
        #             "https://example.com/bird3.jpg"
        #         ]
        #     }

        async with aiohttp.ClientSession() as session:
            tasks = []
            for category, urls in image_urls.items():
                await self.download_images_by_category(session, base_path, category, urls)


if __name__ == '__main__':
    td = TrainsData()
    td.get_sci_db_img(is_save=True)
