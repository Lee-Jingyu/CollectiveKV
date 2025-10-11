# =========================================================================
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
import torch
import fuxictr_version
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
import src as model_zoo
from src.longctr_dataloader import LongCTRDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import gc
import argparse
import os
from pathlib import Path


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--checkpoint', type=str, help='The model checkpoint path.')
    parser.add_argument('--exp_name', type=str, default='baseline', help='The experiment name.')
    parser.add_argument('--method', type=str, default=None, help='Use baseline attention')
    parser.add_argument('--usr_dim', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=10000)
    parser.add_argument('--loss_balance_weight', type=float, default=1.0, help='Weight for loss_balance')
    parser.add_argument('--loss_peak_weight', type=float, default=1.0, help='Weight for loss_peak')
    parser.add_argument('--share_k', type=int, default=1)
    parser.add_argument('--share_v', type=int, default=1)
    args = vars(parser.parse_args())
    
    
    exp_name = args['exp_name']
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['method'] = args['method']
    params['usr_dim'] = args['usr_dim']
    params['pool_size'] = args['pool_size']
    params['loss_balance_weight'] = args['loss_balance_weight']
    params['loss_peak_weight'] = args['loss_peak_weight']
    params['share_k'] = bool(args['share_k'])
    params['share_v'] = bool(args['share_v'])
    params['expid'] = experiment_id
    params['gpu'] = args['gpu']
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    # Build feature_map and transform data
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    
    checkpoint = torch.load(args['checkpoint'], map_location="cuda:0")
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.load_state_dict(checkpoint)

    params["data_loader"] = LongCTRDataLoader
    
    test_result = {}
    if params["test_data"]:
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
        
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    result_filename = f'test_results/MRL_{result_filename}'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(test_result)))
