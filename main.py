import cv2 as cv
import numpy as np

def preprocess_frame(frame):
    """
    Resizes the image, applies grayscale to it and then applies the gaussian blur for smoothing.
    Gaussian Blur Intuition: https://www.youtube.com/watch?v=-LD9MxBUFQo&pp=ygUNZ2F1c3NpYW4gYmx1cg%3D%3D
    """
    stretched_image = cv.resize(frame, (800, 600), interpolation=cv.INTER_AREA)
    grayscale_resized_image = cv.cvtColor(stretched_image, cv.COLOR_BGR2GRAY)
    gaussian_filter_resized_grayscale = cv.GaussianBlur(grayscale_resized_image, (7, 7), 1.2, borderType=cv.BORDER_REPLICATE)
    return gaussian_filter_resized_grayscale

def edge_detector(frame):
    edge_detected_frame = cv.Canny(frame, 40, 175)
    return edge_detected_frame

def region_selector(frame):
    """
    Creates a region mask for the input frame.
    Applies hough transform to generate lines in the parameter space for the image.
    Hough Transform Intuition: https://www.youtube.com/watch?v=XRBc_xkZREg&ab_channel=FirstPrinciplesofComputerVision
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = frame.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.3, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.7, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    
    return cv.bitwise_and(frame, frame, mask=mask)


def hough_transformer(frame): 
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    
    return cv.HoughLinesP(frame, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

def slope_intercept_calculator(lines):
    lines_on_left = []
    length_lines_on_left = []
    lines_on_right = []
    length_lines_on_right = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            
            # Since we use the sintheta, costheta version of a line, a line with negative slope lies of the left side
            if slope < 0:
                lines_on_left.append((slope, intercept))
                length_lines_on_left.append((length))
            if slope > 0:
                lines_on_right.append((slope, intercept))
                length_lines_on_right.append((length))
    
    left_lane = np.dot(length_lines_on_left, lines_on_left) / np.sum(length_lines_on_left) if len(length_lines_on_left) > 0 else None
    right_lane = np.dot(length_lines_on_right, lines_on_right) / np.sum(length_lines_on_right) if len(length_lines_on_right) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, lane_line):
    if lane_line is None:
        return None
    
    slope, intercept = lane_line
    
    x1 = (y1 - intercept) / slope
    x1 = int(x1) if x1 is not None else None
    x2 = (y2 - intercept) / slope
    x2 = int(x2) if x2 is not None else None
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def line_creator(points):
    left_lane, right_lane = slope_intercept_calculator(points)
    
    y1 =  600
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
    

def line_mapper(lines, frame):
    line_color = (255, 0, 0)
    line_thickness = 10
    line_image = np.zeros_like(frame)
    
    for line in lines:
        if line is not None:
            cv.line(line_image, *line, color=line_color, thickness=line_thickness)
    
    return cv.addWeighted(frame, 1.0, line_image, 1.0, 1.0)


def main():
    # video_name = "Road Stock Footage _ Amazing Nature _ Drone _ No Copyright Videos.mp4"
    video_name = "Driving Along An Empty Road _ Royalty Free 4K Stock Video Footage (1).mp4"
    cap = cv.VideoCapture(f"Videos/{video_name}")
    
    if not cap.isOpened():
        print("Couldn't open video file!")
    
    while(cap.isOpened()):
        
        framePresent, frame = cap.read()        
        
        if framePresent:
            preprocessed_frame = preprocess_frame(frame)
            edge_detected_frame = edge_detector(preprocessed_frame)
            region_selected_edges = region_selector(edge_detected_frame)
            hough_transformed_frame = hough_transformer(region_selected_edges)
            detected_lines = line_creator(hough_transformed_frame)
            draw_lane_lines = line_mapper(detected_lines, cv.resize(frame, (800, 600), interpolation=cv.INTER_AREA))
            
            cv.imshow("Lane Detector", draw_lane_lines)
        
            if cv.waitKey(25) & 0xFF == ord("q"):
                break
        
        else:
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()