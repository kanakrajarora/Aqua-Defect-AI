import cv2 
import numpy as np
from ultralytics import YOLO

def Detect_Roughness(image):

# Load image
    img = cv2.imread(image)  # Replace with actual path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Laplacian to detect texture
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Normalize Laplacian for better visibility
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold to isolate rough areas
    _, thresh = cv2.threshold(laplacian_norm, 20, 255, cv2.THRESH_BINARY)

    # Morphological operations (optional but improves contour detection)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on a copy of the original image
    output = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:  # Filter out very small areas
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return output

def Detect_diameter(image):


    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get extreme points
    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

    # Draw the line
    output = image.copy()
    cv2.line(output, leftmost, rightmost, (255, 0, 0), 2)
    cv2.circle(output, leftmost, 5, (0, 255, 0), -1)
    cv2.circle(output, rightmost, 5, (0, 255, 0), -1)
    # Calculate diameter in pixels
    diameter_px = np.linalg.norm(np.array(leftmost) - np.array(rightmost))
    PIXEL_TO_MM = 5.60 / 423.02  # Use your scale
    diameter_mm = diameter_px * PIXEL_TO_MM
    text = f"Diameter: {diameter_mm:.2f} mm"

    # Choose position and draw the text
    cv2.putText(
        output, text,
        (50, 50),  # x, y position (adjust as needed)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,               # font scale
        (0, 255, 255),   # color (BGR - yellow here)
        2,               # thickness
        cv2.LINE_AA      # anti-aliased
    )
    return output

def Detect_material_mix(image):
# Load image
    img = cv2.imread(image)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge and convert back to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # Inverse binary threshold to highlight dark spots
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Morphological operation to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area filtering thresholds (tune as needed)
    min_area = 30   # discard small specks
    max_area = 2000  # discard very large shadows or artifacts

    # Draw filtered contours
    output = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if min_area < area < max_area:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 1)

    return output

def Detect_potholes(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Input image is empty.")
        return None, 0


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   
    kernel_size = (15, 15)


    blackhat_thresh_val = 15

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply the Black-Hat operator. This isolates dark spots on a light background.
    # The result 'blackhat' is an image where potholes appear as bright white spots.
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


    _, mask = cv2.threshold(blackhat, blackhat_thresh_val, 255, cv2.THRESH_BINARY)

    min_area = 1   # The minimum pixel area to be considered a defect. Filters out tiny noise.
    max_area = 60 # The maximum pixel area. Filters out large false positives.
    # ----------------------------------------

    # Find the contours of the white areas in our mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw the results on.
    result_image = image.copy()
    defect_count = 0

    # Loop over all found contours.
    for c in contours:
        # Filter contours based on their area.
        if min_area < cv2.contourArea(c) < max_area:
            defect_count += 1
            # Get the bounding box for the contour.
            (x, y, w, h) = cv2.boundingRect(c)
            # Draw a red rectangle around the detected defect.
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if result_image is not None:
        # Determine the status and color for the text overlay.
        if defect_count > 0:
            status_text = f"Status: DEFECTIVE ({defect_count} potholes found)"
            status_color = (0, 0, 255)  # Red in BGR
        else:
            status_text = "Status: OK"
            status_color = (0, 255, 0)  # Green in BGR

        # Add the status text to the final image.
        cv2.putText(result_image, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    return result_image

def Detect_sunlight(image):
# Load the image
# Replace 'tank_image.jpg' with your image file
    image = cv2.imread(image) 

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and smooth the image.
    # This helps in merging scattered light points ('netting')
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    (T, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # You can print the threshold value 'T' to see what the algorithm chose.
    print(f"Otsu's threshold value: {T}")

    # At this point, 'thresh' is a binary image where defects are white.

    # Create a kernel for morphological operations
    kernel = np.ones((5,5), np.uint8)

    # Perform a morphological closing to fill small gaps in the defects
    closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Perform a morphological opening to remove small noise
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)


    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw on
    output_image = image.copy()
    defect_count = 0

    # Loop over the contours
    for c in contours:
        # Set a minimum area to filter out noise
        min_area = 500 # Adjust this value based on your needs
        
        if cv2.contourArea(c) > min_area:
            defect_count += 1
            # Draw a bounding box around the detected defect
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red box
            
            # You can also draw the contour itself
            # cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2) # Green contour

    return output_image

def Detect_shade(image):
# Load image
    img = cv2.imread(image)  # Change file name as needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Get global mean brightness
    mean_val = np.mean(blurred)

    # Make dark detection more aggressive by increasing the offset
    dark_threshold_upper = int(mean_val - 5)   # Reduced from -15 → more sensitive
    dark_threshold_upper = max(0, dark_threshold_upper)  # Avoid negative

    # Create mask for darker regions
    dark_mask = cv2.inRange(blurred, 0, dark_threshold_upper)

    # Optional: Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Overlay dark regions on image
    highlighted = img.copy()
    highlighted[dark_mask_clean > 0] = [0, 0, 255]  # Mark darker areas in red

    return highlighted
    
def detect_lumps(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found at {image_path}")
        return
    output = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    ###### STEP 1: Bright Lump Detection using Laplacian ######
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)
    _, bright_thresh = cv2.threshold(laplacian_norm, 25, 255, cv2.THRESH_BINARY)
    bright_thresh = cv2.morphologyEx(bright_thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    ###### STEP 2: Dark Rough Patch Detection using Inverted Threshold ######
    _, dark_thresh = cv2.threshold(laplacian_norm, 20, 255, cv2.THRESH_BINARY_INV)
    dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    ###### STEP 3: Contour Detection and Filtering ######
    for mask, color, label in [(bright_thresh, (0, 255, 0), 'Bright lump'), 
                               (dark_thresh, (0, 0, 255), 'Dark speck')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if 400 < area < 2000:
                cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                cv2.putText(output, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return output

def Detect_non_blue_regions(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define blue range in HSV (tune if needed)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create blue mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Invert the mask: now we get non-blue regions
    non_blue_mask = cv2.bitwise_not(blue_mask)

    # Optional: clean up noise
    kernel = np.ones((5, 5), np.uint8)
    non_blue_clean = cv2.morphologyEx(non_blue_mask, cv2.MORPH_OPEN, kernel)
    non_blue_clean = cv2.morphologyEx(non_blue_clean, cv2.MORPH_CLOSE, kernel)

    # Highlight non-blue areas in red
    output = img.copy()
    output[non_blue_clean > 0] = [0, 0, 255]

    return output

def black_lining(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive thresholding to highlight dark lines
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 17, 3
    )

    # Morphological operations to enhance thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of black lines
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    output = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    return output

def mould_joint_mismatch(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=10, threshold2=100)

    # Optional: Find contours on the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    image_with_defects = image.copy()
    cv2.drawContours(image_with_defects, contours, -1, (0, 0, 255), 2)

    # Display results
    return image_with_defects

def joint_cut(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw filtered contours on a copy of the image
    output = image.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # Filter small noise
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    return output

def burn_white_mark_detection(image_path, min_contour_area=50, save_path=None, show=True):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    output = image.copy()

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Rust/Burn Detection
    inverted = cv2.bitwise_not(gray_enhanced)
    rust_thresh = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 10)
    rust_cleaned = cv2.morphologyEx(rust_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_rust, _ = cv2.findContours(rust_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_rust:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)  # Red for rust/burn

    # White Mark Detection
    _, white_thresh = cv2.threshold(gray_enhanced, 240, 255, cv2.THRESH_BINARY)
    white_cleaned = cv2.morphologyEx(white_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_white, _ = cv2.findContours(white_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_white:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)  # Red for white marks too

    return output


def detect_damage(image_path, model_path="damage.pt", conf_threshold=0.25):
    # Load model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Run prediction
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    return image

def improper_weld_finishing(image_path, model_path="weld.pt", conf_threshold=0.25):
    # Load model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Run prediction
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    return image

def detect_blowhole(image_path, model_path="blowhole.pt", conf_threshold=0.25):
    # Load model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Run prediction
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    return image

def detect_blowhole_contamination(image_path, model_path="blowhole_contamination.pt", conf_threshold=0.25):
    # Load model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Run prediction
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    return image
