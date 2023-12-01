import cv2
import math
from scipy import signal
from typing import Tuple, Union
from PIL import Image
from PIL import ImageFilter
from pymatting import *
import numpy as np


def smooth_morphologyEx(image: np.ndarray, ksize_kernel: int = 5, ksize_smooth: int = 5, thresh_min: int = 128,
                        thresh_max: int = 255, interation: int = 3):
    """
    Function will be return "magic" border value for erosion and dilation. It is automatically transformed to Scalar::
    all(-DBL_MAX) for dilation.
    It can perform advanced morphological transformations using an erosion and dilation as basic operations.
    Args:
        image: is an image need to be smoothed. Type np.ndarray
        ksize_kernel: is a kernel of image (averaging filter on an image).take the average, and replace the central
            pixel with the new average value. This operation is continued for all the pixels in the image
        ksize_smooth: is a kernel of image (averaging filter on an image)
        thresh_min: threshold of gaussian blur algorithm
        thresh_max: threshold of gaussian blur algorithm
        interation: interation of morphologyEX
    Returns: np.ndarray

    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_kernel, ksize_kernel))

    (thresh, binRed) = cv2.threshold(image, thresh_min, thresh_max, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=interation)

    image_out = cv2.GaussianBlur(opening, (ksize_smooth, ksize_smooth), 0)

    return image_out


def smooth_polygonelist(image: np.ndarray, thres_min: int = 0, maxval: int = 255):
    """
     Function will be returns a resampled contour, so this will still return a set of (x, y) points. If you want to crop out this result,
        the returned contour should be a N x 1 x 2 NumPy array, so remove the singleton dimension, then do standard
        min/max operations on the x and y coordinates to get the top left and bottom right corners and finally crop
    Args:
        image: is an image need to be smoothed. Type np.ndarray
        thres_min:
        maxval:

    Returns: image

    """
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold (just in case gray is not binary image).
    ret, thresh_gray = cv2.threshold(gray, thres_min, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    canvas = np.zeros(image.shape, np.uint8)

    # creating polygons from contours
    polygonelist = []

    for cnt in contours:
        # define contour approx
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.00003 * perimeter  # 0.005*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        polygonelist.append(approx)

    cv2.drawContours(canvas, polygonelist, -1, (255, 255, 255), 3)

    image = Image.fromarray(canvas)
    return image


def smooth_anti_aliasing(image: np.ndarray, color: Tuple = (255, 255, 255), max_dis_between_point: float = 0.25,
                         window_length: int = 50,
                         poly_order: int = 3, mode: Union[str] = "nearest"):
    """
    I have an image whose edge looks super edgy and blocky. I want to anti-aliasing, but as far as I know, with super
    sampling, I am taking the average color of nearby pixel to make the image looks less jagged and gradient.
    But I don't really want that. I need the output to be curvy, but without the gradient effect.
    The first step is to get the contours for the image. Then, each contour is converted to a list.
    That list is interpolated so that no two successive points are too far apart.
    Finally this list of points is smoothed using scipy.signal.savgol_filter()
    Args:

        mode:
        image: is an image need to be smoothed. Type np.ndarray
        color: is a color to draw (defaut: (255,255,255)
        max_dis_between_point: default :0.25
        window_length:
        poly_order:

    Returns: np.ndarray

    """
    colors = color
    max_dist_between_points = max_dis_between_point
    # Get contours
    gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurs = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    value, thresh = cv2.threshold(blurs, 100, 255, cv2.THRESH_BINARY_INV)

    # edges = cv2.Canny(gray, 123, 123)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def distance(a, b):
        """

        Args:
            a:
            b:

        Returns:

        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def max_dist(points):
        """

        Args:
            points:

        Returns:

        """
        max_dist = 0
        for i in range(len(points) - 1):
            dist = distance(points[i], points[i + 1])
            if dist > max_dist: max_dist = dist
        return max_dist

    def interpolate(points):
        """

        Args:
            points:

        Returns:

        """
        interpolated = [points[0]]
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            dist = distance(a, b)
            if dist >= max_dist_between_points:
                midpoint = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                interpolated.append(midpoint)
            interpolated.append(b)
        return interpolated

    # Iterate through each contour
    for contour in contours:

        # Reformat the contour into two lists of X and Y values
        points, new_points = list(contour), []
        for point in points: new_points.append(tuple(point[0]))
        points = new_points

        # Interpolate the contour points so that they aren't spread out
        while max_dist(points) > 2:
            print(len(points))
            points = interpolate(points)
        X, Y = zip(*points)

        print("size x", X)
        # Define smoothing parameters
        window_length, polyorder = window_length, poly_order
        # Smooth
        X = signal.savgol_filter(X, window_length, polyorder, mode=mode)
        Y = signal.savgol_filter(Y, window_length, polyorder, mode=mode)
        # Re zip and iterate through the points
        smooth = list(zip(X, Y))
        max_a_1, max_b_1 = 0, 0
        min_a_0, min_b_0 = 10000, 10000
        for i in range(len(smooth) - 1):
            if smooth[i][1] > max_a_1:
                max_a_1 = smooth[i][1]
            if smooth[i][1] < min_a_0:
                min_a_0 = smooth[i][1]
            if smooth[i][0] > max_b_1:
                max_b_1 = smooth[i][0]
            if smooth[i][0] < min_b_0:
                min_b_0 = smooth[i][0]

        for point in range(len(smooth)):
            a, b = smooth[point - 1], smooth[point]
            a, b = tuple(np.array(a, int)), tuple(np.array(b, int))
            try:

                blue, green, red = image[b[1], b[0]]

                if b[1] > ((max_a_1 - min_a_0) % 2):
                    blue, green, red = image[b[1] - 1, b[0]]
                elif b[1] < ((max_a_1 - min_a_0) % 2):
                    blue, green, red = image[b[1] + 1, b[0]]

                if b[0] > ((max_b_1 - min_b_0) % 2):
                    blue, green, red = image[b[1], b[0] - 1]
                elif b[0] < ((max_b_1 - min_b_0) % 2):
                    blue, green, red = image[b[1], b[0] + 1]

                color = (int(blue), int(green), int(red))
                cv2.line(image, a, b, tuple(color), 1)
            except:
                pass
    out_image = cv2.GaussianBlur(image, (5, 5), 0)
    return out_image


def smooth_filter(image: np.ndarray):
    image = Image.fromarray(image)
    # Apply SMOOTH filters
    smoothenedImage = image.filter(ImageFilter.ModeFilter(15))
    # moreSmoothenedImage = image.filter(ImageFilter.SMOOTH_MORE)
    image = np.asarray(smoothenedImage)
    return image


def smooth_filter_edge(image: np.ndarray):
    image = Image.fromarray(image)
    # smooth_edge = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    # smooth_edge = image.filter()
    km = np.array((
        (1, 4, 6, 4, 1),
        (4, 16, 24, 16, 4),
        (6, 24, 36, 24, 6),
        (4, 16, 24, 16, 4),
        (1, 4, 6, 4, 1),
    )) / 256.

    smooth_edge = image.filter(
        ImageFilter.Kernel(
            size=km.shape,
            kernel=km.flatten(),
            scale=np.sum(km),  # default
            offset=0)  # defaul
    )

    # smooth_edge = smooth_edge.filter(ImageFilter.CONTOUR)
    image = np.asarray(smooth_edge)
    return image


def smooth_addWeight(image: np.ndarray):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    smooth = cv2.addWeighted(blur, 1, image, 1, 0)
    return smooth


def get_trimap(alpha: np.ndarray):
    # k_size = random.choice(range(2, 5))
    # iterations = np.random.randint(5, 15)
    k_size = 2
    iterations = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


def smooth_trimap_refine_edge(image: np.ndarray, trimap: np.ndarray):
    image = image[..., :3]
    image = np.array(image) / 255.0
    trimap = np.array(trimap) / 255.0
    # estimate alpha from image and trimap
    alpha = estimate_alpha_cf(image, trimap)

    # make gray background
    background = np.zeros(image.shape)
    background[:, :] = [0.5, 0.5, 0.5]

    # estimate foreground from image and alpha
    foreground = estimate_foreground_ml(image, alpha)

    # blend foreground with background and alpha, less color bleeding
    new_image = blend(foreground, background, alpha)

    # save cutout
    cutout = stack_images(foreground, alpha)
    cutout = cv2.GaussianBlur(cutout, (3, 3), 0)
    save_image("lemur_cutout.png", cutout)

    return cutout


def adjust_mask(image: np.ndarray, sigma=0.33):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Use the median value to calculate lower and upper thresholds for Canny
    median = np.median(blurred)
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    # Create a binary mask based on the edges
    mask = np.zeros_like(edges)
    mask[edges > 0] = 255
    # Dilate the mask to cover more area around the edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=dilated_mask)

    return result


def smooth_edge_external(image: np.ndarray, border_size: int = 12, smooth_factor: int = 3):
    """

    Args:
        image: is an image need to be smoothed. Type np.ndarray
        border_size: padding of image
        smooth_factor: factor smooth image

    Returns:

    """
    image = remove_edge_jitter(image, size=5)
    image = down_up(image, 2)
    # Padding the image
    padded_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
    # Smooth the border using a blur effect
    smoothed_image = cv2.GaussianBlur(padded_image, (smooth_factor, smooth_factor), 0)
    # rescale
    smoothed_image = smoothed_image[border_size:-border_size, border_size:-border_size]
    return smoothed_image


def change_segment(image: np.ndarray, segment: np.ndarray, thresh: int = 100):
    """

    Args:
        image:
        segment:

    Returns:

    """
    # Convert the image to grayscale
    print("image type", image, type(image))

    image_ = image[..., :3]

    grayscale = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(grayscale, 100, 200)

    # Find the contour of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (the border)
    border = max(contours, key=cv2.contourArea)

    # Compute the centroid of the border
    M = cv2.moments(border)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Compute the position offset
    dx = cx - int(image.shape[1] // 2)
    dy = cy - int(image.shape[0] // 2)

    # Shift the segment based on the offset
    shifted_segment = segment + np.array([dx, dy])

    return shifted_segment


def remove_edge_jitter(image: np.ndarray, size: int = 30):
    """

    Args:
        image: is an image need to be smoothed. Type np.ndarray
        size: size perform Fast Fourier Transform

    Returns:

    """
    # Perform Fast Fourier Transform (FFT)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Apply High Pass Filter
    rows, cols = image.shape[:2]
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - size:crow + size, ccol - size:ccol + size] = 0

    # Perform Inverse Fast Fourier Transform (IFFT)
    f_ishift = np.fft.ifftshift(fshift)
    denoised_image = np.fft.ifft2(f_ishift)
    denoised_image = np.abs(denoised_image)

    return denoised_image.astype(np.uint8)


def down_up(image: np.ndarray, scale: int = 2):
    """

    Args:
        image: is an image need to be smoothed. Type np.ndarray
        scale: factor resize image

    Returns:

    """
    # Downsample the image
    downsampled = cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR)

    # Upsample the image
    upsampled = cv2.resize(downsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    return upsampled
