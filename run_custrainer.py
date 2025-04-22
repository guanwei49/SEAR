import os
import sys
from copy import deepcopy
from pathlib import Path

from logging import getLogger
from time import time
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from custom_trainer import CustomTrainer  # 导入自定义的Trainer

import csv
from model.sear import SEAR

from seq_dataset import CustomizedSeqDataset

ROOT_DIR = Path(__file__).parent

data_config_dir = os.path.join(ROOT_DIR, 'config', 'data')
model_config_dir = os.path.join(ROOT_DIR, 'config', 'model')



def save_results_to_csv(results, row_name, base_filename='results'):
    metrics = list(next(iter(results.values())).keys()) if results else []

    for metric in metrics:
        filename = f'{base_filename}_{metric}.csv'

        # 检查文件是否已存在，以决定是否写入表头
        try:
            with open(filename, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([''] + list(results.keys()))  # 第一行列名
        except FileExistsError:
            pass  # 文件已存在，不需要重新写入表头

        # 追加数据
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([row_name] + [metrics_dict[metric] for metrics_dict in results.values()])


def create_dataset(config):
    dataset = CustomizedSeqDataset(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def main(model_class, data_config_path, model_config_path):
    config = Config(model=model_class, config_file_list=[data_config_path, model_config_path])
    import torch
    if config['device']==torch.device("cuda"):
        config['device'] = torch.device(f"cuda:{config['gpu_id']}")

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # return

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    test_data.trainData_item_counter = deepcopy(train_data.dataset.item_counter)
    valid_data.trainData_item_counter = deepcopy(train_data.dataset.item_counter)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)


    # trainer loading and initialization
    trainer = CustomTrainer(config, model)  # 使用自定义的Trainer

    training_start_time = time()

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    training_end_time = time()
    #  进行评估
    test_start_time = time()
    grouped_results = trainer.evaluate(test_data, show_progress=config["show_progress"])
    # grouped_results = trainer.evaluate(test_data, show_progress=config["show_progress"],model_file = r'./saved/MY_Freeze-Mar-17-2025_20-34-02')
    test_end_time = time()

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {grouped_results}")

    metrics = list(next(iter(grouped_results.values())).keys()) if grouped_results else []
    grouped_results['training_time(s)'] = {metric: training_end_time-training_start_time for metric in metrics}
    grouped_results['testing_time(s)'] = {metric: test_end_time-test_start_time for metric in metrics}
    save_results_to_csv(grouped_results, row_name=model_class.__name__ + '_' + data_config_path,  base_filename=f'results')



if __name__ == '__main__':
    model_class = SEAR

    model_config_file = (model_class
                         .__name__ + '.yaml')

    data_config_file_list = [ 'beauty.yaml',   'games.yaml', 'supplies.yaml', 'scientific.yaml',  'ml-1m.yaml']

    for data_config_file in data_config_file_list:
        data_config_path = os.path.join(data_config_dir, data_config_file)
        model_config_path = os.path.join(model_config_dir, model_config_file)

        main(model_class, data_config_path, model_config_path)
