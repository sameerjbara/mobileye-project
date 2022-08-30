# This file contains the skeleton you can use for traffic light attention

# Internal imports.. Should not fail
import consts
from misc_goodies import show_image_and_gt

try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: a H*W*3 RGB image of dtype np.uint8 (RGB, 0-255)
    :param kwargs: Whatever you want
    :return: Dictionary with at least the following keys: 'x', 'y', 'col', each containing a list (same lengths)
    # Note there are no explicit strings in the code. ALWAYS USE A CONSTANT VARIABLE INSTEAD!
    """

    # Okay... Here's an example of what this function should return. You will write your own of course
    x_red = (np.arange(-100, 100, 20) + c_image.shape[1] / 2).tolist()
    y_red = [c_image.shape[0] / 2 - 120] * len(x_red)
    x_green = x_red
    y_green = [c_image.shape[0] / 2 - 100] * len(x_red)
    return {consts.X: x_red + x_green,
            consts.Y: y_red + y_green,
            consts.COLOR: [consts.RED] * len(x_red) + [consts.GRN] * len(x_green),
            }


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path), dtype=np.float32) / 255
    if json_path is not None:
        # This code demonstrates the fact you can read the bounding polygons from the json files
        # Then plot them on the image. Try it if you think you want to. Not a must...
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
        ax = show_image_and_gt(image, objects, fig_num)
    else:
        ax = None

    # In case you want, you can pass any parameter to find_tfl_lights, because it uses **kwargs
    attention = find_tfl_lights(image, some_threshold=42)
    tfl_x = np.array(attention[consts.X])
    tfl_y = np.array(attention[consts.Y])
    color = np.array(attention[consts.COLOR])
    is_red = color == consts.RED

    print(f"Image: {image_path}, {is_red.sum()} reds, {len(is_red) - is_red.sum()} greens..")

    # And here are some tips & tricks regarding matplotlib
    # They will look like pictures if you use jupyter, and like magic if you use pycharm!
    # You can zoom one image, and the other will zoom accordingly.
    # I think you will find it very very useful!
    plt.figure()
    plt.clf()
    plt.subplot(211, sharex=ax, sharey=ax)
    plt.imshow(image)
    plt.title('Original image.. Always try to compare your output to it')
    plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=4)
    plt.plot(tfl_x[~is_red], tfl_y[~is_red], 'g+', markersize=4)
    # Now let's convolve.. Cannot convolve a 3D image with a 2D kernel, so I create a 2D image
    # Note: This image is useless for you, so you solve it yourself
    useless_image = np.std(image, axis=2)  # No.. You don't want this line in your code
    highpass_kenel_from_lecture = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) - 1 / 9
    hp_result = sg.convolve(useless_image, highpass_kenel_from_lecture, 'same')
    plt.subplot(212, sharex=ax, sharey=ax)
    plt.imshow(hp_result)
    plt.title('Some useless image for you')
    plt.suptitle("When you zoom on one, the other zooms too :-)")


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    # parser.add_argument('-i', '--image', type=str, help='Path to an image')
    # parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = r'C:\Users\dori\Documents\SNC\data\Selected'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
