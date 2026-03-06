import cv2
import os

# Define paths
raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'

# Create processed directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

# Define supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Walk through the raw data directory
for root, dirs, files in os.walk(raw_data_path):
    for file in files:
        # Check if file is an image
        if file.lower().endswith(image_extensions):
            # Construct full file path
            input_path = os.path.join(root, file)
            
            # Read the image
            img = cv2.imread(input_path)
            
            if img is not None:
                # Resize image to 256x256
                resized_img = cv2.resize(img, (256, 256))
                
                # Create corresponding output path
                relative_path = os.path.relpath(root, raw_data_path)
                output_dir = os.path.join(processed_data_path, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the resized image
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, resized_img)
                
                print(f"Processed: {file}")
            else:
                print(f"Failed to read: {file}")

print('Processing complete!')
