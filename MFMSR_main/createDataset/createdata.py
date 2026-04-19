import os
import argparse
import cv2
from PIL import Image

# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images",
                    action="store_true")
# parser.add_argument('--input_img_dir', type=str, default=r'DOTA',
#                     help='path to high resolution image dir')
parser.add_argument('--hr_img_dir', type=str, default=r'GT',
                    help='path to high resolution image dir')
parser.add_argument('--lr_img_dir', type=str, default=r'LR',
                    help='path to desired output dir for downsampled images')
args = parser.parse_args()

# input_image_dir=args.input_img_dir
hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir

# print(args.input_img_dir)
print(args.hr_img_dir)
print(args.lr_img_dir)
# nums=1
# for file in os.listdir(input_image_dir):
#     im = Image.open(input_image_dir +"/"+ file)
#     out = im.resize((512, 512))
#     out.save(hr_image_dir +"/"+ file)
#     print(nums)
#     nums=nums+1
# create LR image dirs
# os.makedirs(lr_image_dir + "/X2", exist_ok=True)
# os.makedirs(lr_image_dir + "/X3", exist_ok=True)
os.makedirs(lr_image_dir, exist_ok=True)
# os.makedirs(lr_image_dir + "/X6", exist_ok=True)

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")

# Downsample HR images
count=1
for filename in os.listdir(hr_image_dir):
    print(count)
    if not filename.endswith(supported_img_formats):
        continue

    name, ext = os.path.splitext(filename)

    # Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])

    # Blur with Gaussian kernel of width sigma = 1
    hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)
    # cv2.GaussianBlur(hr_img, (0,0), 1, 1)
    # Downsample image 2x
    # lr_image_2x = cv2.resize(hr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)
    #
    # cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + 'x2' + ext), lr_image_2x)

    # Downsample image 3x
    # lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename.split('.')[0] + 'x3' + ext), lr_img_3x)

    # Downsample image 4x
    lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir, filename.split('.')[0] + ext), lr_img_4x)
    count=count+1
    # Downsample image 6x
    # lr_img_6x = cv2.resize(hr_img, (0, 0), fx=1 / 6, fy=1 / 6,
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_6x = cv2.resize(lr_img_6x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/X6", filename.split('.')[0] + 'x6' + ext), lr_img_6x)