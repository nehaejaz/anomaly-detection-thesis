import cv2
import os

def apply_illumination_changes(image_path, output_folder, alpha=7.5, beta=10):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
   
    # List all files in the input folder
    image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Read the image
        input_image_path = os.path.join(image_path, image_file)
        image = cv2.imread(input_image_path)

        if image is not None:
            # Apply contrast and brightness changes
            modified_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # Save the modified image to the output folder
            output_image_path = os.path.join(output_folder, f"{image_file}")
            cv2.imwrite(output_image_path, modified_image)

            print(f"Processed: {input_image_path} -> {output_image_path}")

if __name__ == "__main__":
    # Set your input and output folders
    input_folder = "/home/nejaz/RegAD/utils/original_images/good"
    output_folder = "/home/nejaz/RegAD/MPDD/tubes/test/good"

    # Set the contrast (alpha) and brightness (beta) values
    contrast_factor = 0.5
    brightness_value = 0

    # Apply illumination changes
    apply_illumination_changes(input_folder, output_folder, alpha=contrast_factor, beta=brightness_value)
