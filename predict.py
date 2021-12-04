import argparse
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data._utils.collate import default_collate

from datagen import InpaintingDataset
from modules.ffc import FFCResNetGenerator
from utils.utils import load_config, move_to_device


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--indir', type=str, default='LaMa_test_images')
    parser.add_argument('--outdir', type=str, default='LaMa_test_outputs')
    parser.add_argument('--checkpointdir', type=str,
                        default='pretrained/lama-places/ffc.pth')

    return parser.parse_args()


def main(predict_config_path='configs/prediction/default.yaml'):
    # 加载推理配置文件
    predict_config = load_config(predict_config_path)

    # 重写部分配置
    cli_param = get_parameters()
    predict_config.device = f'cuda:{cli_param.device}'
    predict_config.indir = cli_param.indir
    predict_config.outdir = cli_param.outdir
    predict_config.checkpointdir = cli_param.checkpointdir

    # 模型配置文件
    generator_config = predict_config.model.generator

    kind = generator_config.pop('kind')
    print(f'Make generator {kind}.')

    # 加载模型
    generator = FFCResNetGenerator(**generator_config)

    checkpoint_path = os.path.join(predict_config.model.checkpointdir)
    state = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(state)

    device = torch.device(predict_config.device)
    generator.to(device)
    generator.eval()

    # 读取测试数据
    if not predict_config.indir.endswith('/'):
        predict_config.indir += '/'

    predict_config.dataset.pop('kind')
    dataset = InpaintingDataset(predict_config.indir, **predict_config.dataset)

    # 推理
    with torch.no_grad():
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir,
                os.path.splitext(mask_fname[len(predict_config.indir):])[
                    0] + '.png'
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

            batch = move_to_device(default_collate([dataset[img_i]]), device)
            batch['mask'] = (batch['mask'] > 0) * 1

            # 推理数据组装
            img = batch['image']
            mask = batch['mask']
            masked_img = img * (1 - mask)
            masked_img = torch.cat([masked_img, mask], dim=1)

            # 前向传播
            batch['predicted_image'] = generator(masked_img)

            # 后处理，非mask部分保持原始值
            batch['inpainted'] = mask * batch['predicted_image'] + \
                (1 - mask) * batch['image']

            cur_res = batch[predict_config.out_key][0].permute(
                1, 2, 0).detach().cpu().numpy()

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)


if __name__ == '__main__':
    main()
