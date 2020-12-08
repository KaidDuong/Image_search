import numpy as np
import os
# import the necessary packages
import numpy as np
import cv2
import skimage.io

# class InferenceConfig(GraphicalObjectConfig):
#             # Set batch size to 1 since we'll be running inference on
#             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1

# def detections(ROOT_DIR, image_paths):
#     print(image_paths)
#     MODEL_DIR = os.path.join(ROOT_DIR,'static','model_weights')
#     inference_config = InferenceConfig()

#     # Recreate the model in inference mode
#     model = MaskRCNN(mode="inference", 
#                           config=inference_config,
#                           model_dir=MODEL_DIR)

#     # Get path to saved weights
#     # Either set a specific path or find last trained weights
#     model_path = os.path.join(MODEL_DIR, "mask_rcnn_graphical_object_0080.h5")
#     # model_path = model.find_last()

#     # Load trained weights
#     print("Loading weights from ", model_path)
#     model.load_weights(model_path, by_name=True)
    
#     print("[INFO] Load images ...")
#     # load image 
#     images = load_image(image_paths)
    
#     print('[INFO] Deteting...')
#     results = []
#     for image in images:
#         # Run object detection
#         result = model.detect([image], verbose=1)

#         # save the result
#         results.append(result)
#     print('[INFO] Finish the detection.')
#     return results

def load_image(image_paths):
    """Load the specified images and return a list of the [H,W,3] Numpy array.
    """
    images = []
    for path in image_paths: 
        # Load image
        image = skimage.io.imread(path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        
        images.append(image)
    return images


def dhash(image, hashSize=8):
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# resize the grayscale image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_hash(h):
    # convert the hash to NumPy's 64-bit float and then back to
	# Python's built in int
	return int(np.array(h, dtype="float64"))

def hamming(a, b):
    # compute and return the Hamming distance between the integers
	return bin(int(a) ^ int(b)).count("1")