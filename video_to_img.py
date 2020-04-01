import os
import cv2
import shutil
import imutils
import numpy

video_folder = "video"
output_folder = "output"
ips = 2  # images to be save per second


def extract_to_img(video_file_name):
    video_file_path = "{}/{}".format(video_folder, video_file_name)
    video_name = os.path.splitext(video_file_name)[0]
    video_folder_output = "{}/{}".format(output_folder, video_name)

    if os.path.exists(video_folder_output):
        shutil.rmtree(video_folder_output)
    os.mkdir(video_folder_output)

    vidcap = cv2.VideoCapture(video_file_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    print("Video fps: {}".format(fps))
    print("Images to be saved per second: {}".format(ips))

    count = 0
    frames = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if (count * ips) % fps == 0:
            save_img(image, os.path.join(video_folder_output, "{:s}_f{:d}.jpg".format(video_name, count)))
            save_img(blur_image(image), os.path.join(video_folder_output, "{:s}_f{:d}blur.jpg".format(video_name, count)))
            save_img(noise_image(image), os.path.join(video_folder_output, "{:s}_f{:d}noise.jpg".format(video_name, count)))
            # rotate
            for angle in (5, -5, 10, -10):
                rotated_image = rotate_image(image, angle)
                save_img(rotated_image, os.path.join(video_folder_output, "{:s}_f{:d}rotated{:d}.jpg".format(video_name, count, angle)))
                save_img(blur_image(rotated_image), os.path.join(video_folder_output, "{:s}_f{:d}rotated{:d}blur.jpg".format(video_name, count, angle)))
                save_img(noise_image(rotated_image), os.path.join(video_folder_output, "{:s}_f{:d}rotated{:d}noise.jpg".format(video_name, count, angle)))
            frames += 1
        count += 1
    print("{} frames are extracted to {}.".format(frames, video_folder_output))


def save_img(image, img_name):
    square_img = crop_square_image(image)
    img_224 = resize_to_224(square_img)
    # save frame as JPEG file
    cv2.imwrite(img_name, img_224)


def rotate_image(image, angle):
    # return imutils.rotate_bound(image, angle)
    return imutils.rotate(image, angle)


def blur_image(image):
    return cv2.GaussianBlur(image, (11, 11), 0)


def noise_image(image):
    gauss = numpy.random.normal(0, 0.75, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    return cv2.add(image, gauss)


def flip_image(image):  # horizontal
    return cv2.flip(image, 1)


def crop_square_image(frame):
    # get the Region of Interest (y,y)
    height, width, channel = frame.shape
    if width == height:
        return frame
    midpoint_x = int(width / 2.0)
    midpoint_y = int(height / 2.0)

    if width > height:
        left_x = midpoint_x - midpoint_y
        cropped = frame[0:height, left_x:width - left_x]
        return cropped
    else:
        top_y = midpoint_y - midpoint_x
        cropped = frame[top_y:height - top_y, 0:width]
        return cropped


def resize_to_224(frame):
    frame = crop_square_image(frame)
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return resized


print("OpenCV version: {}".format(cv2.__version__))
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(video_folder):
    for file in f:
        files.append(file)

for f in files:
    print("---------------------------------------")
    print("Extracting video file {} ...".format(f))
    extract_to_img(f)
