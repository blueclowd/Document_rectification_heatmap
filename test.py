import os
import numpy as np
import cv2
import csv
import itertools
import math
from collections import defaultdict
from statistics import mean, stdev
from utils.utils import convert_to_square

from unet import UNET

image_height, image_width = 96, 96
n_keypoints = 5


def findCoordinates(mask):
    '''
    For single key point
    :param mask:
    :return:
    '''
    hm_sum = np.sum(mask)

    index_map = [j for i in range(image_height) for j in range(image_width)]
    index_map = np.reshape(index_map, newshape=(image_height, image_width))

    x_score_map = mask * index_map / hm_sum
    y_score_map = mask * np.transpose(index_map) / hm_sum

    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return px, py


def find_coordinates(heatmap, threshold=1):
    '''
    For multiple key points
    :param heatmap:
    :param threshold:
    :return:
    '''

    ret, thresholded = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(thresholded, np.uint8))

    return centroids[1:]


def group_keypoints(centers, tls, trs, brs, bls, threshold):

    combinations = list(itertools.product(tls, trs, brs, bls))

    centroids = []
    for combination in combinations:
        centroid_x = sum([pt[0] for pt in combination]) / len(combination)
        centroid_y = sum([pt[1] for pt in combination]) / len(combination)

        centroids.append(((centroid_x, centroid_y), combination))

    matches = []
    tl_selected = []
    tr_selected = []
    br_selected = []
    bl_selected = []

    distances = defaultdict(list)
    for center in centers:

        center_x, center_y = center

        for centroid_idx, centroid in enumerate(centroids):

            pred_centroid_x, pred_centroid_y = centroid[0]

            distance = math.sqrt(pow(center_x - pred_centroid_x, 2) + pow(center_y - pred_centroid_y, 2))

            distances[center].append((distance, centroid))

    closest_matches = []

    for center, distance in distances.items():

        distance.sort(key=lambda i: i[0])
        closest_matches.append((distance[0][0], center, distance[0][1][1]))

    closest_matches.sort(key=lambda i: i[0])

    for (distance, center, corners) in closest_matches:

        tl, tr, br, bl = corners

        if distance > threshold or tl in tl_selected or tr in tr_selected or br in br_selected or bl in bl_selected:
            continue
        else:
            matches.append((center, (tl, tr, br, bl)))

            tl_selected.append(tl)
            tr_selected.append(tr)
            br_selected.append(br)
            bl_selected.append(bl)

    return matches


def args_processor():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default="", help="Path to the document image")
    parser.add_argument("-iv", "--videoPath", default="")
    parser.add_argument("-o", "--outputPath", default="", help="Path to store the result")
    parser.add_argument("-r", "--reportPath", default="", help="Path to report file(.csv)")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")
    parser.add_argument("-w", "--weightPath", help="Model for document corners detection", default="")

    return parser.parse_args()


if __name__ == "__main__":

    args = args_processor()

    # Load model
    unet = UNET(input_shape=(image_height, image_width, 1))
    unet.load_weights(args.weightPath)

    # Case 1: Video mode
    if args.videoPath != '':

        cap = cv2.VideoCapture(args.videoPath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_size = max(frame_width, frame_height)

        # Assume A4 aspect ratio
        warped_size = (1200, int(1200 * 1.4142))
        dst_pts = np.float32(
            [[0, 0], [warped_size[0], 0], [warped_size[0], warped_size[1]], [0, warped_size[1]]]).reshape(-1, 1, 2)

        display_patch = np.zeros((target_size, target_size, 3), np.uint8)
        cv2.putText(display_patch, text='Warped 1', org=(650, 400), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=target_size * 0.001, color=(255, 255, 255),
                    lineType=cv2.LINE_AA, thickness=int(target_size * 0.002))

        cv2.putText(display_patch, text='Warped 2', org=(2200, 400), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=target_size * 0.001, color=(255, 255, 255),
                    lineType=cv2.LINE_AA, thickness=int(target_size * 0.002))

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(os.path.join(args.outputPath, 'multiple.avi'), fourcc, 20.0,
                                       (int(target_size * 2), target_size))

        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to square image
            square, padding_dist, padding_axis = convert_to_square(frame, (128, 128, 128))

            img = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (image_width, image_height))
            img = np.float32(img) / 255
            img = np.reshape(img, newshape=(image_height, image_width, 1))

            test_imgs = np.array([img], dtype=np.float32)

            # Prediction
            pred = unet.predict_on_batch(test_imgs)[0]
            mask_pred = np.reshape(pred, newshape=(image_height, image_width, n_keypoints))

            pred_list = []

            scale_x, scale_y = target_size / image_width, target_size / image_height

            for k in range(n_keypoints):

                centroids = find_coordinates(mask_pred[:, :, k], threshold=0.4)

                pred_list.append([(pt[0] * scale_x, pt[1] * scale_y) for pt in centroids])

            prediction = {'tl': np.array(pred_list[0]), 'tr': np.array(pred_list[1]), 'br': np.array(pred_list[2]),
                          'bl': np.array(pred_list[3])}

            tls = pred_list[0]
            trs = pred_list[1]
            brs = pred_list[2]
            bls = pred_list[3]
            cens = pred_list[4]

            groups = group_keypoints(cens, tls, trs, brs, bls, threshold=frame_width * 0.15)

            img_overlay = square.copy()

            # Visualization
            for group in groups:
                centroid, corners = group

                tl, tr, br, bl = corners

                centroid = (int(centroid[0]), int(centroid[1]))
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                cv2.polylines(img_overlay, pts=[np.array([[tl], [tr], [br], [bl]], np.int32)], isClosed=True,
                              color=(255, 255, 255), thickness=max(int(0.005 * frame_width), 10), lineType=cv2.LINE_AA)

            for (x, y) in tls:
                cv2.circle(img_overlay, (int(x), int(y)), radius=int(0.01 * frame_width),
                           color=(255, 255, 0),
                           thickness=-1)
                cv2.putText(img_overlay, text=str(0), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * frame_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * frame_width))

            for (x, y) in trs:
                cv2.circle(img_overlay, (int(x), int(y)), radius=int(0.01 * frame_width),
                           color=(255, 255, 0),
                           thickness=-1)
                cv2.putText(img_overlay, text=str(1), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * frame_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * frame_width))

            for (x, y) in brs:
                cv2.circle(img_overlay, (int(x), int(y)), radius=int(0.01 * frame_width),
                           color=(255, 255, 0),
                           thickness=-1)
                cv2.putText(img_overlay, text=str(2), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * frame_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * frame_width))

            for (x, y) in bls:
                cv2.circle(img_overlay, (int(x), int(y)), radius=int(0.01 * frame_width),
                           color=(255, 255, 0),
                           thickness=-1)
                cv2.putText(img_overlay, text=str(3), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * frame_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * frame_width))

            for (x, y) in cens:
                cv2.circle(img_overlay, (int(x), int(y)), radius=int(0.01 * frame_width),
                           color=(255, 255, 0),
                           thickness=-1)
                cv2.putText(img_overlay, text='G', org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * frame_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * frame_width))

            # Warping

            # Sort by centroid's x-coordinate
            groups.sort(key=lambda g: g[0][0])

            for i, group in enumerate(groups):

                centroid, corners = group

                src_pts = np.float32([corners]).reshape(-1, 1, 2)
                homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                aligned_img = cv2.warpPerspective(square, homography_matrix, warped_size)

                display_patch[600:600 + warped_size[1], 400*(i+1)+warped_size[0]*i:400*(i+1)+warped_size[0]*(i+1), :] = aligned_img

            img_overlay = np.hstack((img_overlay, display_patch))

            video_writer.write(img_overlay)
            img_overlay = cv2.resize(img_overlay, (1024, 512))

            cv2.imshow('frame', img_overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        video_writer.release()

    # Case 2: Image mode
    elif args.imagePath != '':

        test_folder = args.imagePath
        image_names = os.listdir(test_folder)
        image_names = [image_name for image_name in image_names if image_name.endswith(('.jpg', '.png'))]
        image_names.sort()

        ori_imgs = []
        test_imgs = []

        for image_name in image_names:

            img = cv2.imread(os.path.join(test_folder, image_name))
            ori_imgs.append(img.copy())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (image_width, image_height))
            img = np.float32(img) / 255
            img = np.reshape(img, newshape=(image_height, image_width, 1))

            test_imgs.append(img)

        test_imgs = np.array(test_imgs, dtype=np.float32)

        # Prediction
        preds = unet.predict_on_batch(test_imgs)

        predictions = {}
        for i, pred in enumerate(preds):

            print(image_names[i])

            ori_img = ori_imgs[i]
            ori_height, ori_width = ori_img.shape[:2]
            scale_x, scale_y = ori_width / image_width, ori_height / image_height

            mask_pred = np.reshape(pred, newshape=(image_height, image_width, n_keypoints))

            gt_list = []
            pred_list = []

            for k in range(n_keypoints):

                centroids = find_coordinates(mask_pred[:, :, k], threshold=0.4)

                pred_list.append([(pt[0] * scale_x, pt[1] * scale_y) for pt in centroids])

            prediction = {'tl': np.array(pred_list[0]), 'tr': np.array(pred_list[1]), 'br': np.array(pred_list[2]), 'bl': np.array(pred_list[3])}

            predictions[image_names[i]] = prediction

            tls, trs, brs, bls, cens = pred_list

            groups = group_keypoints(cens, tls, trs, brs, bls, threshold=ori_width*0.15)

            for group in groups:
                centroid, corners = group

                tl, tr, br, bl = corners

                centroid = (int(centroid[0]), int(centroid[1]))
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                cv2.polylines(ori_img, pts=[np.array([[tl], [tr], [br], [bl]], np.int32)], isClosed=True,
                             color=(255, 255, 255), thickness=max(int(0.005 * ori_width), 10), lineType=cv2.LINE_AA)

            # Draw model prediction
            for (x, y) in tls:
                cv2.circle(ori_img, (int(x), int(y)), radius=int(0.01 * ori_width),
                          color=(255, 255, 0),
                          thickness=-1)
                cv2.putText(ori_img, text=str(0), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * ori_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * ori_width))

            for (x, y) in trs:
                cv2.circle(ori_img, (int(x), int(y)), radius=int(0.01 * ori_width),
                          color=(255, 255, 0),
                          thickness=-1)
                cv2.putText(ori_img, text=str(1), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                           fontScale=int(0.002 * ori_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                           thickness=int(0.003 * ori_width))

            for (x, y) in brs:
                cv2.circle(ori_img, (int(x), int(y)), radius=int(0.01 * ori_width),
                          color=(255, 255, 0),
                          thickness=-1)
                cv2.putText(ori_img, text=str(2), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                           fontScale=int(0.002 * ori_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                           thickness=int(0.003 * ori_width))

            for (x, y) in bls:
                cv2.circle(ori_img, (int(x), int(y)), radius=int(0.01 * ori_width),
                          color=(255, 255, 0),
                          thickness=-1)
                cv2.putText(ori_img, text=str(3), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                           fontScale=int(0.002 * ori_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                           thickness=int(0.003 * ori_width))

            for (x, y) in cens:
                cv2.circle(ori_img, (int(x), int(y)), radius=int(0.01 * ori_width),
                          color=(255, 255, 0),
                          thickness=-1)
                cv2.putText(ori_img, text='G', org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                           fontScale=int(0.002 * ori_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                           thickness=int(0.003 * ori_width))

            cv2.imwrite(os.path.join(args.outputPath, image_names[i]), ori_img)

        # Load ground truth
        gts = {}
        csv_path = os.path.join(args.imagePath, 'gt.csv')
        with open(csv_path) as csv_file:

            csv_reader = csv.reader(csv_file)

            for line in csv_reader:
                image_name = line[0]
                img = cv2.imread(os.path.join(args.imagePath, image_name))
                img_height, img_width = img.shape[:2]

                gts[line[0]] = {'tl': np.array([float(line[1]), float(line[2])]),
                                'tr': np.array([float(line[3]), float(line[4])]),
                                'br': np.array([float(line[5]), float(line[6])]),
                                'bl': np.array([float(line[7]), float(line[8])])}

        # Evaluation(Root mean square error): Assume single document
        rmses = {}
        n_skips = 0
        for file_name, coordinates in predictions.items():
            gt = gts[file_name]

            # Ignore cases lack of predictions
            if len(coordinates['tl']) == 0 or len(coordinates['tr']) == 0 or len(coordinates['br']) == 0 or len(coordinates['bl']) == 0:
                n_skips += 1
                continue

            rmse = math.sqrt(
                mean(np.concatenate(((coordinates['tl'][0] - gt['tl']) ** 2, (coordinates['tr'][0] - gt['tr']) ** 2,
                                     (coordinates['bl'][0] - gt['bl']) ** 2, (coordinates['br'][0] - gt['br']) ** 2))))

            rmses[file_name] = rmse

        rmse_mean = mean([value for file_name, value in rmses.items()])
        rmse_std = stdev([value for file_name, value in rmses.items()])

        print('Ignore {} samples'.format(n_skips))
        print('RMSR(mean): {}'.format(rmse_mean))
        print('RMSR(std): {}'.format(rmse_std))

        with open(args.reportPath, 'w', newline='') as csv_file:

            csv_writer = csv.writer(csv_file)
            for file_name, rmse in rmses.items():
                csv_writer.writerow([file_name, rmse])

            csv_writer.writerow(['Mean', rmse_mean])
            csv_writer.writerow(['Std', rmse_std])

        print('Generated RMSE report.')
