import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure


def get_regex_alphabetique_simple(verbose=0):
    pattern = re.compile(r'[^a-zA-Z]')
    return pattern

def disk_structure(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)


def granulometry(data, sizes=None):
    s = max(data.shape)
    if sizes is None:
        sizes = range(1, s/2, 2)
    granulo = [ndimage.binary_opening(data, \
            structure=disk_structure(n)).sum() for n in sizes]
    return granulo

def get_img(img_path):
    return Image.open(img_path)

def get_img_black_white(img):
    res = None
    if img is not None:
        if isinstance(img, str):
            res = get_img_black_white(get_img(img))
        elif isinstance(img, Image.Image):
            res =img.convert('L')
    return res

def get_gaussian_img(im_grey, n = 10, l = 256):
    im = None
    if im_grey is not None:
        if isinstance(im_grey, str):
            im_grey = get_img_black_white(im_grey)
        elif isinstance(im_grey, Image.Image):
            im = ndimage.gaussian_filter(im_grey, sigma=l/(4.*n))
    return im

def resize_picture(img, scale_percent=60, verbose=0):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def define_img_point(img_src_path, resize_scale_percent=None, display_cv2=False, display=True, nb_descriptors=300, verbose=0):
    #reading the image using imread() function from cv2 module and converting it into gray image
    grayimage = cv2.imread(img_src_path,0)
    if resize_scale_percent is not None:
        grayimage = resize_picture(grayimage, scale_percent=resize_scale_percent, verbose=verbose)
        
    #grayimage = cv2.cvtColor(readimage, cv2.COLOR_BGR2GRAY)
    #creating a sift object and using detectandcompute() function to detect the keypoints and descriptor from the image
    equ = cv2.equalizeHist(grayimage)

    siftobject = cv2.xfeatures2d.SIFT_create(nb_descriptors)
    keypoint, descriptor = siftobject.detectAndCompute(equ, None)
    #drawing the keypoints and orientation of the keypoints in the image and then displaying the image as the output on the screen
    keypointimage = cv2.drawKeypoints(equ, keypoint, grayimage, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if display:
        if display_cv2:
            cv2.imshow('SIFT', keypointimage)
            cv2.waitKey()
        else:
            plt.imshow(keypointimage)
            plt.title(f'img_kp size: {keypointimage.shape} - Gray: {grayimage.shape}') 
            plt.axis('off')
            plt.show()
    return grayimage, keypoint, descriptor


def transform_kp(file, scale_percent=None):
    gray= cv2.imread(file,0)
    if scale_percent is not None:
        gray = resize_picture(gray, scale_percent=scale_percent)
    equ = cv2.equalizeHist(gray)
    sift = cv2.SIFT_create(300)
    kp, des = sift.detectAndCompute(equ,None)
    img_kp =cv2.drawKeypoints(equ,kp,gray, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg',img_kp)
    print(f'Descripteurs :  {des.shape}')
    # print(des)

    # cv2.imshow('figure',img_kp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(img_kp)
    plt.title(f'img_kp size: {img_kp.shape} - Gray: {gray.shape}') 
    plt.axis('off')
    plt.show()
    return kp, des, img_kp

# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
def display_img_Gaussian(im, src_img=None, im_grey=None):
    mask = im > im.mean()
    label_im, _ = ndimage.label(mask)

    plt.figure(figsize=(9,3))

    sub = 131
    if src_img is not None:
        sub = 141
        if im_grey is not None:
            sub = 151
        plt.subplot(sub)
        plt.imshow(src_img)
        plt.axis('off')
        sub += 1
    else:
        if im_grey is not None:
            sub = 141

    if im_grey is not None:
        plt.subplot(sub)
        plt.imshow(im_grey)
        plt.axis('off')
        sub += 1

    plt.subplot(sub)
    plt.imshow(im)
    plt.axis('off')

    sub+=1
    plt.subplot(sub)
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.axis('off')

    sub+=1
    plt.subplot(sub)
    plt.imshow(label_im, cmap=plt.cm.nipy_spectral)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
    plt.show()

def display_granulometry(im):
    mask = im > im.mean()

    granulo = granulometry(mask, sizes=np.arange(2, 19, 4))

    plt.figure(figsize=(6, 2.2))

    plt.subplot(121)
    plt.imshow(mask, cmap=plt.cm.gray)
    opened = ndimage.binary_opening(mask, structure=disk_structure(10))
    opened_more = ndimage.binary_opening(mask, structure=disk_structure(14))
    plt.contour(opened, [0.5], colors='b', linewidths=2)
    plt.contour(opened_more, [0.5], colors='r', linewidths=2)
    plt.axis('off')
    plt.subplot(122)
    plt.plot(np.arange(2, 19, 4), granulo, 'ok', ms=8)

    plt.subplots_adjust(wspace=0.02, hspace=0.15, top=0.95, bottom=0.15, left=0, right=0.95)
    plt.show()


def display_greyscale_dilation(im_grey):
    dist = ndimage.distance_transform_bf(im_grey)
    dilate_dist = ndimage.grey_dilation(dist, size=(3, 3), \
            structure=np.ones((3, 3)))

    bigger_points = ndimage.grey_dilation(im_grey, size=(5, 5), structure=np.ones((5, 5)))

    plt.figure(figsize=(12.5, 3))
    plt.subplot(141)
    plt.imshow(im_grey, interpolation='nearest', cmap=plt.cm.nipy_spectral)
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(bigger_points, interpolation='nearest', cmap=plt.cm.nipy_spectral)
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(dist, interpolation='nearest', cmap=plt.cm.nipy_spectral)
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(dilate_dist, interpolation='nearest', cmap=plt.cm.nipy_spectral)
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.show()


def display_errosion(im_grey):
    open_square = ndimage.binary_opening(im_grey)
    eroded_square = ndimage.binary_erosion(im_grey)
    reconstruction = ndimage.binary_propagation(eroded_square, mask=im_grey)

    plt.figure(figsize=(9.5, 3))
    plt.subplot(131)
    plt.imshow(im_grey, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(open_square, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(reconstruction, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.show()


def show_histogramme(img_param, equalize=False):
    dst = None
    img_l =img_param
    if isinstance(img_l, list) == False:
        img_l = [img_param]

    nb_graph = len(img_l)
    sub = int("2"+str(nb_graph)+"0")
    if equalize:
        sub = int(str(nb_graph)+"40")

    fig = plt.figure(figsize=(15, 5*nb_graph))

    for img in img_l:
        sub = _draw_hist_img(img=img, sub=sub, fig=fig)
        if equalize:
            dst = cv2.equalizeHist(img)
            sub = _draw_hist_img(img=dst, sub=sub, fig=fig)
    plt.show()
    return dst


def show_hog(img_path, reduce_ratio=None, cmap="BrBG",orientations=9, pixels_per_cell=(8, 8)):
    img = imread(img_path)
     
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title(f'Original : {img.shape}')

    if reduce_ratio is not None:
        # resizing image
        resized_img = resize(img, (img.shape[0]/reduce_ratio, img.shape[1]/reduce_ratio))
    else:
        resized_img = img.copy()
    
    #creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    ax3.axis('off')
    ax3.imshow(hog_image, cmap=cmap)
    ax3.set_title(f'Hog Image : {hog_image.shape}')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=cmap)
    ax2.set_title(f'Hog Rescaled : {hog_image_rescaled.shape}')
    plt.show()
    # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    return img, resized_img, hog_image




def _draw_hist_img(img, sub, fig):
    sub += 1
    ax = fig.add_subplot(sub)
    ax.imshow(img)
    ax.axis('off')
    sub += 1

    ax = fig.add_subplot(sub)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax.plot(cdf_normalized, color = 'b')
    ax.hist(img.flatten(),256,[0,256], color = 'r')
    ax.set_xlim([0,256])
    ax.legend(('cdf','histogram'), loc = 'upper left')
    return sub

def _draw_hist_img_cv2(img, sub, fig):
    sub += 1
    cv2.imshow('figure',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sub += 1

    ax = fig.add_subplot(sub)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax.plot(cdf_normalized, color = 'b')
    ax.hist(img.flatten(),256,[0,256], color = 'r')
    ax.set_xlim([0,256])
    ax.legend(('cdf','histogram'), loc = 'upper left')
    return sub