"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob

import cv2
import numpy as np

def binarize(img):
    '''
    calcuates the optimal  thresold by maximising between 
    class varinace itrating over an histogram of the image
    '''
    vals,hist = binning(img)
    total = hist.sum()
    sumb = 0
    wb = 0
    maximum = 0
    level = 128
    sum1 = sum(vals*hist)
    for i in range(256):
        wf = total - wb
        if wb > 0 and wf > 0:
            mf = (sum1 - sumb)/wf
            val = wb * wf * ((sumb / wb) - mf) * ((sumb / wb) - mf)
            if val >= maximum:
                level = i
                maximum = val
        wb = wb + hist[i]
        sumb = sumb + (i-1)*hist[i] 
    return (img > level-2)*255

def binning(img,bins=256):
    '''
    this function bins the pixels by their intensity and returns a histogram of the image
    '''
    x,y = img.shape
    hist = np.zeros(bins)
    for i in range(x):
        for j in range(y):
            hist[int(img[i,j])] += 1
    return list(range(bins)),hist

def crop(arr):
    '''
    crops the input image by finding the boundries of the character
    '''
    x,y = arr.shape
    xtop,xbot = 0,x
    for i in range(x-1):
        if arr[i].all() != arr[i+1].all():
            if arr[i].all():
                xtop = i+1
            else:
                xbot = i+1
    yleft,yright = 0,y
    for i in range(y-1):
        if arr[:,i].all() != arr[:,i+1].all():
            if arr[:,i].all():
                yleft = i+1
            else:
                yright = i+1
    return arr[xtop:xbot,yleft:yright]

def scale(img, rows, cols):
    '''
    resizes the images to the desired resolution and 
    outputs a binarized image of the resized image
    '''
    R,C = img.shape
    out = [[ img[int(R * r / rows)][int(C * c / cols)] for c in range(cols)] for r in range(rows)]
    return np.array(out)

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    processed_chars = enrollment(characters)

    boxes = detection(test_img)
    
    results = recognition(boxes,test_img,processed_chars)

    print("results saved")

    return results

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    targets = [item[0] for item in characters]

    dis_file="./features/intermediate_"+"".join(sorted(targets)) +".json"

    processed = dict()
    if os.path.isfile(dis_file):
        with open(dis_file,'r') as json_file:
            processed = json.load(json_file)

    for target,img in characters:
        if target not in processed.keys():
            processed[target] =scale(crop(binarize(img)),256,256).tolist()

    with open(dis_file,'w') as fp:
        json.dump(processed,fp,indent=4,sort_keys=True)

    print("enrollment done")

    return processed


def detection(image):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    image = binarize(image)
    breaks = []

    # splitting the image into smaller parts by splitting into individual lines of text
    for i in range(image.shape[0]-1):
        if (not image[i].all() and image[i+1].all()) or (image[i].all() and not image[i+1].all()):
            breaks.append(i+1)
    lines = [image[:breaks[0]]]
    for i in range(len(breaks)-1):
        lines.append(image[breaks[i]:breaks[i+1]])
    lines.append(image[breaks[-1]:])
    
    final_out = []
    x = 1
    archives = []
    conflicts = 0
    order = []
    r = 0 # row tracker

    '''
    intreating over lines of text to label groups of pixels 
    to resolve conflicts between two adjacent groups.
    This results in the final bounding box of each character
    '''
    for img in lines:
        archive = {}
        ref = np.zeros(img.shape,dtype=np.uint32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if not img[i][j]:
                    up,left = None,None
                    if 0<=i-1<img.shape[0] and ref[i-1][j]:
                        up = ref[i-1][j]
                    if 0<=j-1<img.shape[1] and ref[i][j-1]:
                        left = ref[i][j-1]
                    if up == left == None:
                        ref[i][j] = x
                        archive[x] = [j,i+r,j+1,i+r+1]
                        x += 1
                    elif up == left:
                        ref[i][j] = left
                        a = archive[left]
                        archive[left] = [min(a[0],j),min(a[1],i+r),max(a[2],j+1),max(a[3],i+r+1)]
                    elif up and left:
                        ref[i][j] = up
                        # ref[ref == left] = up
                        a = archive[up]
                        b = archive[left]
                        c = [min(a[0],b[0],j),min(a[1],b[1],i+r),max(a[2],b[2],j+1),max(a[3],b[3],i+r+1)]
                        archive[up] = c
                        ref[:,c[0]:c[2]][ref[:,c[0]:c[2]] == left] = up
                        del archive[left]
                        conflicts += 1 
                    elif left:
                        ref[i][j] = left
                        a = archive[left]
                        archive[left] = [min(a[0],j),min(a[1],i+r),max(a[2],j+1),max(a[3],i+r+1)]
                    elif up:
                        ref[i][j] = up
                        a = archive[up]
                        archive[up] = [min(a[0],j),min(a[1],i+r),max(a[2],j+1),max(a[3],i+r+1)]
                        
        
        if archive:
            archives.append(archive)           
        r += i+1
        final_out.append(ref)

    out = []
    for archive in archives:
        for val in archive.values():
            out.append([val[0],val[1],val[2]-val[0],val[3]-val[1]])
    
    print('detection done')
    return out

def recognition(boxes,test_img,characters,threshold=8500):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    out = []
    test_img = binarize(test_img)
    for box in boxes:
        img = test_img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        img = scale(crop(img),256,256)
        minum = (float('inf'),None)
        for target,target_img in characters.items():
            score = (abs(img - target_img)/255).sum()
            if minum[0] > score:
                minum = (score,target)
        if minum[0] > threshold:
            minum = (minum[0],'UNKNOWN')
        predict =  {"bbox":box,"name":minum[1]}
        out.append(predict)

    print("recognition done")
    return out



# def save_results(coordinates, rs_directory):
#     """
#     Donot modify this code
#     """
#     results = []
#     with open(os.path.join(rs_directory, 'results.json'), "w") as file:
#         json.dump(results, file)

def save_results(results, rs_directory):
    """
    Donot modify this code
    """
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
