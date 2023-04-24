from elliot.run import run_experiment
import gdown
import os
import shutil

print('MODELS:')
print(u'''
+-----------------------------------------------------------------------------------+----------+--------------------------------------------------------+
|                                                 Paper                             |   Name   |                          Link                          |
+-----------------------------------------------------------------------------------+----------+--------------------------------------------------------+
| Neural Graph Collaborative Filtering                                              | ngcf     | https://dl.acm.org/doi/10.1145/3331184.3331267         |
| Disentangled Graph Collaborative Filtering                                        | dgcf     | https://dl.acm.org/doi/abs/10.1145/3397271.3401137     |
| LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation   | lightgcn | https://dl.acm.org/doi/10.1145/3397271.3401063         |
| Self-supervised graph learning for recommendation                                 | sgl      | https://doi.org/10.1145/3404835.3462862                |
| UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation | ultragcn | https://dl.acm.org/doi/10.1145/3459637.3482291         |
| How Powerful is Graph Convolution for Recommendation?                             | gfcf     | https://dl.acm.org/doi/abs/10.1145/3459637.3482264     |
+-----------------------------------------------------------------------------------+----------+--------------------------------------------------------+
''')

while True:
    model = input('Insert model name (ngcf, dgcf, lightgcn, sgl, ultragcn, gfcf): ').lower()
    if model in ['ngcf', 'lightgcn', 'dgcf', 'sgl', 'ultragcn', 'gfcf']:
        break
    else:
        print('Sorry, the model should be one of these: ngcf, dgcf, lightgcn, sgl, ultragcn, gfcf')

print('DATASETS:')
print(u'''
+----------------------+--------+--------+--------------+
|       Dataset        | Users  | Items  | Interactions |
+----------------------+--------+--------+--------------+
| Gowalla              | 29,858 | 40,981 |    1,027,370 |
| Yelp 2018            | 31,668 | 38,048 |    1,561,406 |
| Amazon Book          | 52,643 | 91,599 |    2,984,108 |
+----------------------+--------+--------+--------------+
''')

while True:
    dataset = input('Insert dataset name (gowalla, yelp 2018, amazon book): ').lower()
    if dataset in ['gowalla', 'yelp 2018', 'amazon book']:
        if dataset == 'gowalla':
            if not os.path.exists('data/gowalla'):
                os.makedirs('data/gowalla')
                gdown.download_folder("https://drive.google.com/drive/folders/1j-9g_XXZ3nmgcatePduy-Y0x4KPApJDm?usp=share_link",
                                      use_cookies=False)
                files_list = os.listdir('gowalla')
                for files in files_list:
                    shutil.move('gowalla/' + files, 'data/gowalla/')
                os.rmdir('gowalla')
        elif dataset == 'yelp 2018':
            dataset = 'yelp-2018'
            if not os.path.exists('data/yelp-2018'):
                os.makedirs('data/yelp-2018')
                gdown.download_folder("https://drive.google.com/drive/folders/1pMzJqkhyKM9n8JsyoVFrxYMQzi2PTDoL?usp=share_link",
                                      use_cookies=False)
                files_list = os.listdir('yelp-2018')
                for files in files_list:
                    shutil.move('yelp-2018/' + files, 'data/yelp-2018/')
        else:
            dataset = 'amazon-book'
            if not os.path.exists('data/amazon-book'):
                os.makedirs('data/amazon-book')
                gdown.download_folder("https://drive.google.com/drive/folders/1uHyqJ8KD3DD7IA7V7HzlvvFf3RfRRayJ?usp=share_link",
                                      use_cookies=False)
                files_list = os.listdir('amazon-book')
                for files in files_list:
                    shutil.move('amazon-book/' + files, 'data/amazon-book/')
                os.rmdir('amazon-book')
        break
    else:
        print('Sorry, the dataset should be one of these: gowalla, yelp 2018, amazon book')

print('\n\n')
run_experiment(f"config_files/{model}_{dataset}.yml")
