import cv2, numpy as np, pickle, time
from PIL import Image
import matplotlib.pyplot as plt


# Epiline drawing
def drawlines(img1,img2,lines,pts1,pts2):
    """
    img1 - image on which we draw the epilines for the points in img2 
    lines - corresponding epilines
    references: https://medium.com/@dc.aihub/3d-reconstruction-with-stereo-images-part-3-epipolar-geometry-98b75e40f59d
    """
    r, c = img1.shape[:2]
    if len(img1.shape)==2: img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)==2: img1 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color  = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1   = cv2.line(img1, (x0,y0), (x1,y1), color, 5)
        img1   = cv2.circle(img1,tuple(pt1),20,color,-1)
        img2   = cv2.circle(img2,tuple(pt2),20,color,-1)
    # for
    return img1, img2
# drawlines

# local feature drawing
def plot_features(im, locations, circle=False):
    # Show image with features. input: im (image as array),locs (row, col, scale, orientation of each feature).
    def draw_circle(c, r):
        t = np.arange(0,1.01,.01)*2*np.pi
        x = int(r*np.cos(t) + c[0])
        y = int(r*np.sin(t) + c[1])
        cv2.circle(im, (x, y), r, (0, 0, 255), 2)
    # draw_circle
    for p in locations:
        if circle:
            cv2.circle(im, tuple(p[:2].astype(np.int)), int(p[2]), (255, 0, 0), 2)
        else:
            cv2.circle(im, tuple(p[:2].astype(np.int)), 2, (255, 0, 0), -1)
    # for
    return im
# plot_features

def process_feature_image(image):
    if hasattr(process_feature_image, "detector")==False:
        # cv2.FastFeatureDetector_create(), cv2.ORB_create(), cv2.xfeatures2d.SIFT_create, cv2.xfeatures2d_DAISY.create()
        process_feature_image.detector = cv2.xfeatures2d.SIFT_create()
    # if
    keypoints, descriptions = process_feature_image.detector.detectAndCompute(image, None)
    return keypoints, descriptions
# feature_process_image

def match_two_image(desc_model, desc_current):
    if hasattr(match_two_image, "matcher")==False:
        match_two_image.matcher = cv2.BFMatcher()
    # if
    matches = match_two_image.match(desc_model, desc_current)
    match_two_image.matches = matches
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
# match_two_image

def match_two_image_c2(desc_model, desc_current):
    if hasattr(match_two_image, "matcher")==False:
        match_two_image_c2.matcher = cv2.BFMatcher()
    # if
    matches = match_two_image_c2.matcher.knnMatch(desc_model, desc_current, k = 2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: good.append([m])
    return good
# match_two_image_c2

def match_twosided_c2(desc_model, desc_current):
    matches_12 = match_two_image_c2(desc_model, desc_current) # query, train
    matches_21 = match_two_image_c2(desc_current, desc_model) # query, train
    
    dict_matches = {}
    for match in matches_21:
        destIdx  = match[0].queryIdx
        modelIdx = match[0].trainIdx
        dict_matches[modelIdx] =  destIdx
    # for
    
    good = []
    for match in matches_12:
        destIdx  = match[0].trainIdx
        modelIdx = match[0].queryIdx
        if dict_matches.get(modelIdx) is not None and dict_matches[modelIdx] == destIdx:
            good.append(match)
    
    return good
# match_two_image

def get_matches_pts(matches, kp_src, kp_dst):
    # differenciate between source points and destination points
    src_pts = np.float32([kp_src[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp_dst[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)
    return src_pts, dst_pts
# get_match_points

def appendimages(im1,im2):
    """
        Return a new image that appends the two images side-by-side.
    """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1,im2), axis=1)
# appendimages

def plot_matches(im1, im2, locs1, locs2, matches, show_below=True):
    """
        Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations), 
        matchscores (as output from ’match()’),
        show_below (if images should be shown below matches).
    """
    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3,im3))
    # if
    cols1 = im1.shape[1]
    for match in matches:
        src_idx = match[0].queryIdx
        dst_idx = match[0].trainIdx
        
        x1, y1 = int(locs1[src_idx][0]), int(locs1[src_idx][1])
        x2, y2 = int(locs2[dst_idx][0] + cols1), int(locs2[dst_idx][1])
        cv2.line(im3, (x1, y1), (x2, y2), (255, 0, 0), 5)
    # for
    return im3
# plot_matches