import cv2
import numpy as np

# Load pre-trained model and points
prototxt_path = "pretrained model/colorization_deploy_v2.prototxt"
model_path = "pretrained model/colorization_release_v2.caffemodel"
kernel_path = "points/pts_in_hull.npy"

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# Prepare the network
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

def colorize_image(image_path, output_path="colorized_image.jpg"):
    # Read the image
    bw_image = cv2.imread(image_path)
    if bw_image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert to LAB color space
    normalized_image = bw_image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2LAB)
    
    # Resize and extract L channel
    resized_image = cv2.resize(lab, (224, 224))
    L = cv2.split(resized_image)[0]
    L = L - 50  # Subtract 50 for mean-centering
    
    # Set the input and get the predicted 'ab' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize the predicted 'ab' channels to match the original image size
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    
    # Get the L channel from the original image
    L = cv2.split(lab)[0]
    
    # Combine the L channel with the predicted 'ab' channels
    colorized = cv2.merge([L, ab])
    
    # Convert back to BGR color space
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    
    # Clip and convert to 8-bit
    colorized = np.clip(colorized * 255, 0, 255).astype(np.uint8)
    
    # Save and display the result
    cv2.imwrite(output_path, colorized)
    
    # Display the result
    cv2.imshow("Original", bw_image)
    cv2.imshow("Colorized", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return colorized

# Example usage:
if __name__ == "__main__":
    # Replace with your image path
    input_image = "lion.jpg"
    colorize_image(input_image)
