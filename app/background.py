import backgroundremover.utilities
import backgroundremover.bg
from backgroundremover.u2net import detect
import numpy as np
import cv2
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = backgroundremover.bg.get_model("u2net")

def bg_predict(net, item):
    sample = detect.preprocess(item)
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(
                sample["image"].unsqueeze(0).cuda().float()
            )
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())
        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        pred = d1[:, 0, :, :]
        predict = detect.norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        img = predict_np * 255

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

        return img

def remove_background(img):
    mask = bg_predict(model, img)
    # .convert("L")
    mask = mask.astype(np.uint8)
    resized_mask = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]))
    # apply mask to image
    masked = cv2.bitwise_and(img, img, mask=np.array(resized_mask))
    thresh = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    # convert to CV_8UC1
    gray = np.array(gray, dtype=np.uint8)
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    extreme_points = np.array([contours[0][0][0], contours[0][0][0], contours[0][0][0], contours[0][0][0]])

    for contour in contours:
        contour = np.squeeze(contour, axis=1)
        contour_min_x, contour_min_y = np.min(contour, axis=0)
        contour_max_x, contour_max_y = np.max(contour, axis=0)
        
        extreme_points[0] = np.minimum(extreme_points[0], [contour_min_x, contour_min_y])
        extreme_points[1] = np.maximum(extreme_points[1], [contour_max_x, contour_max_y])
        extreme_points[2] = np.minimum(extreme_points[2], [contour_min_x, contour_min_y])
        extreme_points[3] = np.maximum(extreme_points[3], [contour_max_x, contour_max_y])
    
    # crop 
    cropped = masked[extreme_points[0][1]:extreme_points[1][1], extreme_points[2][0]:extreme_points[3][0]]
    return cropped