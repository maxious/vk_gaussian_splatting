import os
import cv2
import numpy as np
import json
import yaml
import sys
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F

def get_gaussian_reconstruction(llrm_root, json_path, output_folder, checkpoint_path, height=540, fix_aspect_ratio=True, input_num=32, sample_method="uniform"):
    sys.path.append(llrm_root)
    from model.llrm import LongLRM

    if height != 540:
        raise NotImplementedError("Only height=540 is supported currently.")
    if input_num != 32:
        raise NotImplementedError("Only input_num=32 is supported currently.") 
    default_config = os.path.join(llrm_root, "configs/7m1t_tm.yaml")
    config = os.path.join(llrm_root, "configs/dl3dv_i540_32input_8target.yaml")
    default_config = yaml.safe_load(open(default_config, 'r'))
    config = yaml.safe_load(open(config, 'r'))
    def recursive_merge(dict1, dict2):
        for key, value in dict2.items():
            if key not in dict1:
                dict1[key] = value
            elif isinstance(value, dict):
                dict1[key] = recursive_merge(dict1[key], value)
            else:
                dict1[key] = value
        return dict1
    default_config = recursive_merge(default_config, config)
    config = default_config
    # remove training related configs
    if 'training' in config:
        for key in list(config['training'].keys()):
            config['training'].pop(key)
    config = edict(config)

    # load data
    scene_data = json.load(open(json_path, 'r'))
    scene_name = scene_data['scene_name']
    frames = scene_data['frames']
    num_frames = len(frames)
    assert num_frames >= input_num, f"Not enough frames: {num_frames} < {input_num}"
    height_orig, width_orig = frames[0]['h'], frames[0]['w']
    aspect_ratio = width_orig / height_orig
    assert aspect_ratio >= 1.0 and aspect_ratio <= 960 / 540, f"Unsupported aspect ratio: {aspect_ratio}"
    if fix_aspect_ratio:
        width = 960
    else:
        width = int(height * aspect_ratio)
    # round width and height to be multiple of patch size
    patch_size = config.model.patch_size
    height = int(round(height / patch_size) * patch_size)
    width = int(round(width / patch_size) * patch_size)

    sample_idxs = []
    if sample_method == "uniform":
        sample_idxs = np.linspace(0, num_frames - 1, input_num, dtype=int)
    else:
        raise NotImplementedError(f"Unsupported sample method: {sample_method}")
    print(f"Sampled frame indices: {sample_idxs}")
    
    input_images = []
    for idx in sample_idxs:
        frame = frames[idx]
        img_path = os.path.join(os.path.dirname(json_path), frame['file_path'])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        image = image.astype(np.float32) / 255.0
        input_images.append(image)
    input_images = np.stack(input_images, axis=0) # (V, H, W, 3)
    input_images = torch.from_numpy(input_images).permute(0,3,1,2).unsqueeze(0) # (1, V, 3, H, W)

    fxs = np.array([frames[idx]['fx'] for idx in sample_idxs]) # (V,)
    fys = np.array([frames[idx]['fy'] for idx in sample_idxs])
    cxs = np.array([frames[idx]['cx'] for idx in sample_idxs])
    cys = np.array([frames[idx]['cy'] for idx in sample_idxs])
    height_ratio = height / height_orig
    width_ratio = width / width_orig
    fxs *= width_ratio
    fys *= height_ratio
    cxs *= width_ratio
    cys *= height_ratio
    input_intr = np.stack([fxs, fys, cxs, cys], axis=1) # (V, 4)
    input_intr = torch.from_numpy(input_intr).unsqueeze(0) # (1, V, 4)

    w2cs = np.stack([np.array(frames[idx]['w2c'], dtype=np.float32) for idx in sample_idxs]) # (V, 4, 4)
    c2ws = np.linalg.inv(w2cs) # (V, 4, 4)
    input_c2ws = torch.from_numpy(c2ws) # (V, 4, 4)

    # normalize poses
    position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
    forward_avg = input_c2ws[:, :3, 2].sum(0) # (3,)
    down_avg = input_c2ws[:, :3, 1].sum(0) # (3,)
    # gram-schmidt process
    forward_avg = F.normalize(forward_avg, dim=0)
    right_avg = F.normalize(torch.cross(down_avg, forward_avg, dim=0), dim=0)
    down_avg = F.normalize(torch.cross(forward_avg, right_avg, dim=0), dim=0)
    pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
    pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)
    pos_avg_inv = torch.inverse(pos_avg)
    input_c2ws = pos_avg_inv.unsqueeze(0) @ input_c2ws
    position_max = input_c2ws[:, :3, 3].abs().max()
    scene_scale = 1.35 * position_max
    scene_scale = 1.0 / scene_scale
    input_c2ws[:, :3, 3] *= scene_scale
    if torch.isnan(input_c2ws).any() or torch.isinf(input_c2ws).any():
        print("encounter nan or inf in input poses")
        return {}
    input_c2ws = input_c2ws.unsqueeze(0) # (1, V, 4, 4)

    input_dict = {
        'input_intr': input_intr,
        'input_c2ws': input_c2ws,
        'input_images': input_images,
    }
    print("Input data prepared.")
    for k, v in input_dict.items():
        print(f"{k}: {v.shape}, dtype: {v.dtype}")

    # load model
    device = torch.device('cuda')
    model = LongLRM(config, device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    status = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    input_dict = {k: v.to(device) for k, v in input_dict.items()}

    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
        ret_dict = model(input_dict)

    # save reconstruction outputs
    video_save_path = os.path.join(output_folder, f"input_traj.mp4")
    gaussian_save_path = os.path.join(output_folder, f"gaussians.ply")
    gaussian_dict = ret_dict['gaussians']
    for key in gaussian_dict:
        if isinstance(gaussian_dict[key], torch.Tensor):
            gaussian_dict[key] = gaussian_dict[key][0]
    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
        model.save_input_video(input_intr[0].to(device), input_c2ws[0].to(device), gaussian_dict, height, width, video_save_path, insert_frame_num = 16)
        print(f"Saved input trajectory video to {video_save_path}")
        model.save_gaussian_ply(gaussian_dict, gaussian_save_path, opacity_threshold=0.004)
        print(f"Saved gaussian splatting reconstruction to {gaussian_save_path}")

    return ret_dict

if __name__ == "__main__":
    llrm_root = "/path/to/Long-LRM"
    json_path = "/path/to/opencv_cameras.json"
    llrm_output_folder = "/path/to/output/folder"
    checkpoint_path = "/path/to/checkpoint.pt"
    os.makedirs(llrm_output_folder, exist_ok=True)
    ret_dict = get_gaussian_reconstruction(llrm_root, json_path, llrm_output_folder, checkpoint_path, height=540, fix_aspect_ratio=False, 
                                           input_num=32, sample_method="uniform")