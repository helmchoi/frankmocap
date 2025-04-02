# Copyright (c) Facebook, Inc. and its affiliates.

# Hyelim Choi, 2025.02
# only images accepted

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    output_dir = args.input_path + "/result"
    args.out_dir = output_dir
    print("out_dir = ", args.out_dir)

    #Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    # assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0

    fps = 0.0
    fps_cnt = 0
    joints3d_stack = []
    joints2d_stack = []

    while True:
        # load data
        load_bbox = True

        # if input_type =='image_dir':
        if cur_frame < len(input_data):
            image_path = input_data[cur_frame]
            img_original_bgr  = cv2.imread(image_path)
        else:
            img_original_bgr = None

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        # bbox detection
        if load_bbox:
            # hand_bbox_list = [ dict(right_hand = np.array([390.0, 50.0, 500.0, 620.0])) ]      # width_start, height_start, width, height
            # hand_bbox_list = [ dict(right_hand = np.array([460.0, 180.0, 530.0, 520.0])) ]      # width_start, height_start, width, height
            hand_bbox_list = [ dict(right_hand = np.array([300.0, 80.0, 660.0, 640.0])) ]
        else:            
            # Input images has other body part or hand not cropped.
            # Use hand detection model & body detector for hand detection
            # assert args.crop_type == 'no_crop'
            detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            _, _, hand_bbox_list, _ = detect_output

        if len(hand_bbox_list) < 1:
            print(f"No hand deteced: {image_path}")
            continue
    
        # Hand Pose Regression
        if cur_frame == 1:  # warmup
            for _ in range(5):
                _ = hand_mocap.regress(
                    img_original_bgr, hand_bbox_list, add_margin=True)
        t0 = time.time()
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        t1 = time.time()
        print("==> inference time: ", t1-t0, ", fps: ", 1/(t1-t0))
        fps += 1.0 / (t1 - t0)
        fps_cnt += 1
        assert len(hand_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize mesh -----------------------------------------
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)

        # save the image (we can make an option here)
        # if args.out_dir is not None:
        demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # SIDE VIEW
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list, side_view=True)
        demo_utils.save_res_img(args.out_dir, image_path[:-4]+"_side.png", res_img)
        # --------------------------------------------------------

        # save predictions
        assert len(pred_output_list) == 1  # batch size = 1
        joints3d = []
        joints2d = []
        for joint in pred_output_list[0]['right_hand']['pred_joints_smpl']:
            joints3d += [float(joint[0]), float(joint[1]), float(joint[2])]
        for joint in pred_output_list[0]['right_hand']['pred_joints_img']:
            joints2d += [float(joint[0]), float(joint[1])]
        joints3d_stack += [joints3d]
        joints2d_stack += [joints2d]

        print(f"Processed : {image_path}")
        
    # #save images as a video
    # if not args.no_video_out and input_type in ['video', 'webcam']:
    #     demo_utils.gen_video_out(args.out_dir, args.seq_name)
    
    print("** Mean FPS: ", fps / float(fps_cnt))
    with open(output_dir + "/pred_joints3d.json", "w") as f:
        json.dump(joints3d_stack, f)
    with open(output_dir + "/pred_joints2d_img.json", "w") as f:
        json.dump(joints2d_stack, f)

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
