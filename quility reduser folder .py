import cv2
import os

def reduce_image_quality(input_image_path, output_image_path, quality):
    # Read the image
    image = cv2.imread(input_image_path)
    
    # Set JPEG quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Save the image with the specified quality
    cv2.imwrite(output_image_path, image, encode_param)
    print(f"Image saved at {output_image_path} with quality {quality}")

def process_images_in_directory(input_dir, output_dir, quality):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        input_image_path = os.path.join(input_dir, filename)

        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            output_image_path = os.path.join(output_dir, filename)
            reduce_image_quality(input_image_path, output_image_path, quality)
        else:
            print(f"Skipped non-image file: {filename}")

# Example usage
input_directory = './cropped/1'
output_directory = './cropped_low_quality/1'
quality = 10  # Quality value (0 to 100), lower means more compression and lower quality

process_images_in_directory(input_directory, output_directory, quality)
