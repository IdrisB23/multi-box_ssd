from pathlib import Path
import cv2
import torch
import numpy as np
import os
import argparse
import math

from itertools import combinations

from ssd import build_ssd

from data import VOC_CLASSES as labels

import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utility_functions import create_output_dir_if_req, clean_output_dir


INTERSECTION_OF_TOTAL_AREA_THRESHOLD = 0.3

BASE_DIR = Path('..')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector')

parser.add_argument('--orig_frames_path', default='sampled_frames/orig',
                    type=str, help='Path to the extracted frames directory')
parser.add_argument('--depth_frames_path', default='sampled_frames/depth',
                    type=str, help='Directory where depth maps of original frames are')
parser.add_argument('--output_path', default='sampled_frames/ssd',
                    type=str, help='Path to where the processed frames should be saved')
parser.add_argument('--pretrained_model_path', default='pre-trained_models/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Pre-trained model path')

args = parser.parse_args()


def aggreg_depth_values_in_ROI(whole_depth_map: np.ndarray, ROI: tuple):
    x1, y1, x2, y2 = ROI[0], ROI[1], ROI[2], ROI[3]
    depth_in_ROI = whole_depth_map[y1:y2, x1:x2]
    return cv2.mean(depth_in_ROI)[0]

def get_corresponding_depth_map(depth_frames_path: Path, index: int) -> np.ndarray:
    depth_map_of_orig_path = depth_frames_path / '{:03d}.png'.format(index+1)
    return cv2.imread(depth_map_of_orig_path.as_posix(), cv2.IMREAD_UNCHANGED)

def calc_area_bb(x1, y1, x2, y2) -> int:
    side1_dist = abs(x2 - x1)
    side2_dist = abs(y2 - y1)
    return side1_dist * side2_dist

def calc_area_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
    # test whether top-left corner of bb1 inside other box, equivalent to whether bottom-right of other inside bb1
    test1topL_x = x11 >= x21 and x11 <= x22 # x11 in [x21, x22]
    test1topL_y = y11 >= y21 and y11 <= y22 # y11 in [y21, y22]
    test1topL = test1topL_x and test1topL_y

    # test whether bottom-right corner of bb1 inside other box, equivalent to whether top-left of other inside bb1
    test1bottomR_x = x12 >= x21 and x12 <= x22 # x12 in [x21, x22]
    test1bottomR_y = y12 >= y21 and y12 <= y22 # y12 in [y21, y22]
    test1bottomR = test1bottomR_x and test1bottomR_y

    if test1topL and test1bottomR:
        raise Exception('bb1 is contained inside the other, shouldn\'t happen')

    if not test1topL and not test1bottomR:
        return 0
    
    if test1bottomR_y:
        side1 = abs(x12 - x21)
        side2 = abs(y12 - y21)
        return side1 * side2
    
    if test1topL:
        side1 = abs(x11 - x22)
        side2 = abs(y11 - y22)
        return side1 * side2


def process_frames_via_mbssd(input_frames_path: Path, depth_frames_path: Path, 
                             output_path: Path, model_path: str):
    # ORIG_FRAMES_DIR = Path(input_frames_path)
    # PROCESSED_FRAMES_DIR = Path(output_path)

    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_weights(model_path)

    # iterate over all imgs within orig frames dir
    for m, p_ in enumerate(input_frames_path.glob('*.png')):
        # ORIG_FRAME_AS_DIR = PROCESSED_FRAMES_DIR / p_.name[:-4]
        # os.mkdir(ORIG_FRAME_AS_DIR)
        image = cv2.imread(str(p_), cv2.IMREAD_COLOR)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = x.unsqueeze(0)
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(image.shape[1::-1]).repeat(2)
        # dict that maps each person's index to their bounding box coordinates
        person_2_boundBox = dict()
        k = 0 # persons detections counter
        for i in range(detections.size(1)):
            j = 0 # detections couter
            while detections[0,i,j,0] >= 0.6:
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                if label_name != 'person':
                    continue
                k += 1

                display_txt = '%s: %.2f'%(label_name, score)

                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                # coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                x1, y1, x2, y2 = int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])

                person_2_boundBox[k] = (x1, y1, x2, y2)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, display_txt, (x1, y1), font, 1, cv2.LINE_AA)
                cv2.imwrite((output_path / p_.name).as_posix(), image)

                # save all cropped persons detected in the img
                cropped_img = image[y1:y2, x1:x2]
                orig_as_dir_name = output_path / p_.name[:-4]
                orig_as_dir_name.mkdir(exist_ok=True)
                out_p_ = (orig_as_dir_name / '{:03d}.png'.format(k)).as_posix()
                cv2.imwrite(out_p_, cropped_img)
                j += 1

        pair_size = 2
        for (bb1, bb2) in list(combinations(range(1, k+1), pair_size)):
            bb1 = person_2_boundBox[bb1]
            print('bb1:', bb1)
            bb2 = person_2_boundBox[bb2]
            print('bb2:', bb2)
            area_bb1 = calc_area_bb(*bb1)
            area_bb2 = calc_area_bb(*bb2)
            area_intersect = calc_area_intersect(*bb1, *bb2)
            print(m, 'area_bb1:', area_bb1, 'area_bb2:', area_bb2, 'area_intersect:', area_intersect)
            if area_intersect > 0.4 * area_bb1:
                print('bb1 is negligeable')
                x1, y1, x2, y2 = bb2
            elif area_intersect > 0.4 * area_bb2:
                print('bb2 is negligeable')
                x1, y1, x2, y2 = bb1
            else:
                print(f'Undecidable by intersection {INTERSECTION_OF_TOTAL_AREA_THRESHOLD}. Using depth map...')
                depth_map_of_orig = get_corresponding_depth_map(depth_frames_path, m)

                aggregate_depth_val_bb1 = aggreg_depth_values_in_ROI(depth_map_of_orig, bb1)
                aggregate_depth_val_bb2 = aggreg_depth_values_in_ROI(depth_map_of_orig, bb2)
                # print(p_.name, '->', aggregate_depth_val_bb1, aggregate_depth_val_bb2)
                if aggregate_depth_val_bb1 > aggregate_depth_val_bb2:
                    x1, y1, x2, y2 = bb1
                elif aggregate_depth_val_bb1 < aggregate_depth_val_bb2:
                    x1, y1, x2, y2 = bb2
                else:
                    print('Also can\'t decide by aggregated depth value, choosing based on bounding box size')
                    if area_bb1 > area_bb2:
                        x1, y1, x2, y2 = bb1
                    elif area_bb1 < area_bb2:
                        x1, y1, x2, y2 = bb2
                    else:
                        print('At this point choosing bb1 randomly')
                        x1, y1, x2, y2 = bb1

            cropped_img = image[y1:y2, x1:x2]
            orig_as_dir_name = output_path / p_.name[:-4]
            orig_as_dir_name.mkdir(exist_ok=True)
            out_p_ = (orig_as_dir_name / 'selected_cropped_img.png').as_posix()
            cv2.imwrite(out_p_, cropped_img)
        print('---/---')



if __name__ == '__main__':
    orig_frames_path = BASE_DIR / args.orig_frames_path
    depth_frames_path = BASE_DIR / args.depth_frames_path
    output_path = BASE_DIR / args.output_path
    pretrained_model_path = args.pretrained_model_path
    create_output_dir_if_req(output_path)
    clean_output_dir(output_path)
    process_frames_via_mbssd(orig_frames_path, 
                             depth_frames_path, 
                             output_path, 
                             pretrained_model_path)