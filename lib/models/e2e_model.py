### yu yang #######
import math
import os
import joblib
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.parse_config import *
from lib.core.config import VIBE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
import numpy as np
from  tqdm import tqdm
import cv2
import time
from lib.utils.yolo_utils import (
    build_targets, to_cpu, non_max_suppression,
    weights_init_normal, load_classes, select_iou_GT)

from lib.models.mpt import MPT as MPT_YU
from multi_person_tracker import MPT
from lib.dataset.inference import Inference
from torch.utils.data import DataLoader
from lib.models.yolov3 import Darknet

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
from lib.utils.renderer import Renderer
import colorsys
import shutil
from lib.models.spin import Regressor, hmr


def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        self.gru.flatten_parameters()
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y

class FeatExtract(nn.Module):
    """
    SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(FeatExtract, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def feature_extractor(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        return xf

class Bottleneck(nn.Module):
    """
    Redefinition of Bottleneck residual block
    Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output

class e2e_VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(e2e_VIBE, self).__init__()
        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.hmr = hmr()
        self.regressor = Regressor()



    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        # feature = self.featExt.feature_extractor(input.reshape(-1, nc, h, w))
        # with torch.no_grad():
        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))
        feature = feature.reshape(batch_size, seqlen, -1)
        feature = self.encoder(feature)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output

class Det_Track_VIBE(nn.Module):
    def __init__(self):
        super(Det_Track_VIBE, self).__init__()
        self.Tracker = MPT(
            device='cuda',
            batch_size=12,
            display=False,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416,
        )
        self.generator = e2e_VIBE(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).cuda()

    def inferData(self, image_path, output_path, pretrained):

        img0 = cv2.imread(osp.join(image_path, os.listdir(image_path)[0]))
        num_frames = len(os.listdir(image_path))
        img_height, img_width = img0.shape[0:2]
        tracking_start_time = time.time()
        tracking_results = self.Tracker(image_path)
        tracking_end_time = time.time()

        print(f'Tracking fps: {(num_frames)/(tracking_end_time - tracking_start_time): .2f}')

        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < 25:
                del tracking_results[person_id]
        ckpt = torch.load(pretrained)
        ckpt = ckpt['gen_state_dict']
        self.generator.load_state_dict(ckpt, strict=False)
        self.generator.eval()

        vibe_results = {}
        vibeFPS = 0
        for person_id in tqdm(list(tracking_results.keys())):
            person_start_time = time.time()
            joints2d = None
            bboxes = tracking_results[person_id]['bbox']
            frames = tracking_results[person_id]['frames']
            dataset = Inference(
                image_folder=image_path,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=1.0,
            )
            bboxes = dataset.bboxes
            frames = dataset.frames
            dataloader = DataLoader(dataset, batch_size=12, num_workers=16)
            with torch.no_grad():
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
                for batch in dataloader:
                    batch = batch.unsqueeze(0)
                    batch = batch.cuda()
                    batch_size, seqlen = batch.shape[:2]
                    # output = model(batch)[-1]
                    output = self.generator(batch)[-1]
                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:, :, 3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :, 75:].reshape(batch_size * seqlen, -1))
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
            vibeFPS += person_frame/person_time

            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=img_width,
                img_height=img_height
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
        joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))
        print(f'Mean target fps: {vibeFPS/len(list(tracking_results.keys()))}')

        return vibe_results

    def renderRes(self, image_path, output_path, vibe_results):
        img0 = cv2.imread(osp.join(image_path, os.listdir(image_path)[0]))
        orig_height, orig_width = img0.shape[0:2]
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

        output_img_folder = f'{image_path}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        num_frames = len(os.listdir(image_path))
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_path, x)
            for x in os.listdir(image_path)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in range(len(image_file_names)):
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
        save_name = 'vibe_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)


class Det_VIBE(nn.Module):
    def __init__(self, cfg):
        super(Det_VIBE, self).__init__()
        self.detector = Darknet(cfg.YOLO.MODEL_DEF)
        self.detector.apply(weights_init_normal)
        self.detector.load_darknet_weights(cfg.YOLO.PRETRAINED_MODEL)
        self.device = cfg.DEVICE
        self.generator = e2e_VIBE(
            seqlen=cfg.DATASET.SEQLEN,
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
        ).to(cfg.DEVICE)
        self.tracker = MPT(
            device=cfg.DEVICE,
            batch_size=cfg.TRACKER.TRACKER_BATCH_SIZE,
            output_format='dict',
        )
        self.nms = non_max_suppression
        self.class_names = load_classes(cfg.YOLO.CLASS_PATH)


    def forward(self, img_batch, gt_bbox_batch=None):
        tmp_shape = img_batch.shape
        img_batch_yolo = img_batch.view([-1, tmp_shape[2], tmp_shape[3],tmp_shape[4]])
        tmp_shape = gt_bbox_batch.shape
        gt_bbox_yolo = gt_bbox_batch.view([-1, tmp_shape[2]])
        self.detector.cuda()
        if gt_bbox_batch is None:
            self.detector.eval()
            with torch.no_grad():

                yolo_output = self.detector(img_batch_yolo)

                from lib.utils.vis import batch_vis_yolo_raw, batch_vis_yolo_res
                # batch_vis_yolo_raw(img_batch_yolo, yolo_output, 0)
                ## apply nms
                yolo_output = self.nms(yolo_output, 0.3, 0.4)

                batch_vis_yolo_res(img_batch_yolo, yolo_output, self.class_names)
                ## apply sort tracker
                tracking_results = self.tracker.run_tracker_pred(yolo_output)
                for person_id in list(tracking_results.keys()):
                    if tracking_results[person_id]['frames'].shape[0] < 25:
                        del tracking_results[person_id]
                ## gen all bbox input
                for person_id in tqdm(list(tracking_results.keys())):
                    joints2d = None
                    bboxes = tracking_results[person_id]['bbox']
                    frames = tracking_results[person_id]['frames']
        else:
            self.detector.eval()
            yolo_output = self.detector(img_batch_yolo)
            select_iou_GT(yolo_output, gt_bbox_yolo, 0.5)
            print('111')








