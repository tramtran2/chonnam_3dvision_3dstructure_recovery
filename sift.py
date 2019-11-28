from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
    # Process an image and save the results in a file. """
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
        # cmmd = str("sift" + imagename + " --output="+resultname+" "+params)
        cmmd = str("sift" + " <" + imagename + " >" + resultname)
        print(cmmd)
        os.system(cmmd)
    print('processed', imagename, 'to', resultname)
# process_image

def read_features_from_file(filename):
    # Read feature properties and return in matrix form.
    # f = np.loadtxt(filename)
    # return f[:,:4],f[:,4:] # feature locations, descriptors
    with open(filename, "rt") as f: # read all lines
        lines = f.readlines()
        float_ints = []
        for line in lines: # at every line, split and convert float
            float_ints.extend([float(x) for x in line.split(" ") if x !=""])
        # for
    # with
    nrows     = int(float_ints[0]) # number of key points
    nfeatures = int(float_ints[1]) # number of features
    float_ints = np.array(float_ints[2:]) # remove two first numbers
    float_ints = float_ints.reshape(nrows, 4 + nfeatures) # reshape (nrows, 4 position + 128 features)
    return float_ints[:, :4], float_ints[:, 4:].astype(dtype=np.int) # feature locations, descriptors
# read_features_from_file

def write_features_to_file(filename,locs,desc):
# Save feature location and descriptor to file. """
    np.savetxt(filename,np.hstack((locs,desc)))

def plot_features(im,locs,circle=False):
    # Show image with features. input: im (image as array),locs (row, col, scale, orientation of each feature).
    def draw_circle(c, r):
        t = np.arange(0,1.01,.01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x,y,'b',linewidth=2)
    # draw_circle
    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2][::-1], p[2])
    else:
        plt.plot(locs[:, 1], locs[:, 0], 'ob')
    plt.axis('off')
# plot_features

def match(desc1, desc2):
    """ 
    For each descriptor in the first image, select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for second image). 
    """
    
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    matchscores = np.zeros((desc1_size[0],1),'int')
    desc2t = desc2.T # precompute matrix transpose
    
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:],desc2t) # vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))
        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores
# match

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

def match_twosided(desc1,desc2):
    """ 
        Two-sided symmetric version of match(). 
    """
    print("Matching Image1 with Image2")
    matches_12 = match(desc1, desc2)
    print("Matching Image2 with Image1")
    matches_21 = match(desc2, desc1)
    
    ndx_12 = matches_12.nonzero()[0]
    # remove matches that are not symmetric
    print("Remove matches that are not symmetric")
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    return matches_12
# match_twosided

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
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
    plt.imshow(im3)
    cols1 = im1.shape[1]

    for i, m in enumerate(matchscores):
        if m>0:
            x1, y1 = locs1[i][1], locs1[i][0]
            x2, y2 = locs2[m[0]][1] + cols1, locs2[m[0]][0]
            plt.plot([x1,x2],[y1, y2],'c')
            # print(i, m[0])
            # print(i, "-->", locs1[i][1], locs1[i][0])
            # print(m[0], "-->", locs2[m[0]][1] + cols1, locs2[m[0]][0])
        # if
    # for
    plt.axis("off")
# plot_matches