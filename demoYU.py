# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo, VIBE_w_HMR
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker
from lib.models.e2e_model import e2e_VIBE

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
    img_folder_Info,
)

MIN_NUM_FRAMES = 25

def runDemo(image_folder, output_folder, pretrained, tracker_batch_size=12, vibe_batch_size=450, wireframe=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    output_path = os.path.join(output_folder, os.path.basename(image_folder).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)
    num_frames, img_shape  = img_folder_Info(image_folder)
    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]
    total_time = time.time()
    # ========= Run tracking ========= #
    bbox_scale = 1.0
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=tracker_batch_size,
        display=False,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=416,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define VIBE model ========= #
    model = e2e_VIBE(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = pretrained
    ckpt = torch.load(pretrained_file)
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    time_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        person_start_time = time.time()
        joints2d = None
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )
        bboxes = dataset.bboxes
        frames = dataset.frames
        dataloader = DataLoader(dataset, batch_size=vibe_batch_size, num_workers=16)
        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
            for batch in dataloader:
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]
                # output = model(batch, J_regressor=J_regressor)[-1]
                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        person_end_time = time.time()

        person_time = person_end_time - person_start_time
        person_frame = len(frames)
        print(f'Person Time: {person_time:.2f}, Person FPS:{person_frame/person_time: .2f} ')



        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    joblib.dump(vibe_results, os.path.join(output_folder, "vibe_output.pkl"))

    # ========= Render results as a single video ========= #
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=wireframe)

    output_img_folder = f'{image_folder}_output'
    os.makedirs(output_img_folder, exist_ok=True)

    print(f'Rendering output video, writing frames to {output_img_folder}')

    # prepare results for rendering
    frame_results = prepare_rendering_results(vibe_results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mc = mesh_color[person_id]

            mesh_filename = None
            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

    # ========= Save rendered video ========= #
    vid_name = os.path.basename(image_folder)
    save_name = 'vibe_result.mp4'
    save_name = os.path.join(output_folder, save_name)
    print(f'Saving result video to {save_name}')
    images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
    shutil.rmtree(output_img_folder)

    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--pretrained', type=str,
                        help='pretrained model path')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    args = parser.parse_args()



    runDemo(args.image_folder, args.output_folder, args.pretrained)
