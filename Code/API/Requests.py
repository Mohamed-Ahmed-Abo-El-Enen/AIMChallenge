import pandas as pd
import requests
from requests.packages import urllib3
import json
from os.path import join
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


url = 'https://recruitment.aimtechnologies.co/ai-tasks'


def get_tweets_api(ids_list):
    return requests.post(url,
                         headers={'Content-Type': 'application/json'},
                         data=json.dumps(ids_list),
                         verify=False).json()


def get_dataset_df(df, save_directory_path="Dataset"):
    len_df = len(df)
    count = 0
    id_list = []
    dialect_list = []
    text_list = []
    while len_df > 0:
        num_samples = min(1000, len_df)
        end_index = count+num_samples-1
        ids_list = list(map(str, df.loc[count: end_index, "id"].values))
        json_dataset = get_tweets_api(ids_list)
        id_list.extend(ids_list)
        dialect_list.extend(list(df.loc[count: end_index, "dialect"].values))
        text_list.extend(json_dataset.values())
        count = end_index
        len_df -= num_samples

    res_df = pd.DataFrame(list(zip(id_list, dialect_list, text_list)),
                          columns=['Id', 'Dialect', "Text"])
    csv_file = join(save_directory_path, "csv_text_dataset.csv")
    res_df.to_csv(csv_file, index=False, encoding='utf-8')
