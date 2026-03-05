import cv2
import numpy as np



class CalculateScene:
    """
    Handles scene analysis including classification, shift detection, and visualization.
    """

    def estimate_ball_size_from_player(self, player_contour, frame_shape, actual_ball_diameter_pixels=None):
        """
        Estimate the golf ball size in pixels based on the detected player's height in the frame.
        Assumes average player height and real golf ball diameter.
        Prints the estimated ball size in pixels.
        Args:
            player_contour: Contour (numpy array) of the detected player
            frame_shape: Shape tuple (height, width) of the frame
        """
        # Constants
        AVERAGE_PLAYER_HEIGHT_M = 1.85  # meters
        GOLF_BALL_DIAMETER_M = 0.043  # meters (43 mm)

        if player_contour is None or len(player_contour) == 0:
            return None

        # Get bounding box of player
        x, y, w, h = cv2.boundingRect(player_contour)
        player_height_pixels = h
        if player_height_pixels == 0:
            return None

        # Estimate pixel-to-meter ratio
        pixels_per_meter = player_height_pixels / AVERAGE_PLAYER_HEIGHT_M
        # Ball size in pixels
        ball_diameter_pixels = GOLF_BALL_DIAMETER_M * pixels_per_meter
        if actual_ball_diameter_pixels is not None:
            diff = ball_diameter_pixels - actual_ball_diameter_pixels
        return ball_diameter_pixels

    def __init__(self):
        pass

    def classify_frame(self, frame):
        """
        Classifies a video frame into one of three categories: sky, grass, or audience.
        Also returns probability percentages for each frame type.
        
        Args:
            frame: Input frame/image (BGR format from OpenCV)
        
        Returns:
            dict: Contains classification and probabilities for each frame type
                  {
                      "classification": "grass",  # "sky", "grass", or "audience"
                      "probabilities": {
                          "sky": 15.5,
                          "grass": 45.2,
                          "audience": 39.3
                      }
                  }
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Define HSV ranges for different frame types
        from config import SKY_LOWER, SKY_UPPER, GRASS_LOWER, GRASS_UPPER, AUDIENCE_LOWER, AUDIENCE_UPPER, CLASSIFY_SKY_MIN_PERCENT, CLASSIFY_GRASS_MIN_PERCENT
        sky_mask = cv2.inRange(hsv, SKY_LOWER, SKY_UPPER)
        sky_pixels = cv2.countNonZero(sky_mask)
        grass_mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
        grass_pixels = cv2.countNonZero(grass_mask)
        audience_mask = cv2.inRange(hsv, AUDIENCE_LOWER, AUDIENCE_UPPER)
        audience_pixels = cv2.countNonZero(audience_mask)
        total_pixels = frame.shape[0] * frame.shape[1]

        # Calculate percentages
        sky_percentage = (sky_pixels / total_pixels) * 100
        grass_percentage = (grass_pixels / total_pixels) * 100
        audience_percentage = (audience_pixels / total_pixels) * 100

        # --- Blue/White sky split ---
        # Blue sky: high blue channel, moderate S, moderate V
        # White sky: high V, low S, blue channel not dominant
        blue_sky_mask = ((h > 90) & (h < 130) & (s > 30) & (v > 100)).astype(np.uint8)
        blue_sky_pixels = cv2.countNonZero(blue_sky_mask)
        blue_sky_percentage = (blue_sky_pixels / total_pixels) * 100
        # White sky: high V, low S
        white_sky_mask = ((s < 30) & (v > 180)).astype(np.uint8)
        white_sky_pixels = cv2.countNonZero(white_sky_mask)
        white_sky_percentage = (white_sky_pixels / total_pixels) * 100

        # Classify based on dominant color
        if sky_percentage > CLASSIFY_SKY_MIN_PERCENT:
            # Further split sky into blue/white
            if blue_sky_percentage > white_sky_percentage:
                classification = "blue_sky"
            else:
                classification = "white_sky"
        elif grass_percentage > CLASSIFY_GRASS_MIN_PERCENT:
            classification = "grass"
        elif audience_percentage < 5 and grass_percentage > 5:
            # If audience is very low and some grass is present, treat as grass
            classification = "grass"
        else:
            classification = "audience"
        #print(f"Classification: {classification} | Sky: {sky_percentage:.2f}%, Blue Sky: {blue_sky_percentage:.2f}%, White Sky: {white_sky_percentage:.2f}%, Grass: {grass_percentage:.2f}%, Audience: {audience_percentage:.2f}%")
        # Return classification and probabilities
        return {
            "classification": classification,
            "probabilities": {
                "sky": round(sky_percentage, 2),
                "blue_sky": round(blue_sky_percentage, 2),
                "white_sky": round(white_sky_percentage, 2),
                "grass": round(grass_percentage, 2),
                "audience": round(audience_percentage, 2)
            }
        }

    def draw_frame_type_indicator(self, frame, classification_result):
        """
        Draws a colored square in the lower left corner of the frame to indicate
        the classified frame type.
        
        Args:
            frame: The video frame (BGR format)
            classification_result: Dictionary returned from classify_frame()
        
        Returns:
            frame: Modified frame with the colored indicator square
        """
        # Define colors for each frame type (BGR format)
        from config import FRAME_INDICATOR_COLORS, FRAME_INDICATOR_SIZE, FRAME_INDICATOR_MARGIN

        classification = classification_result["classification"]
        color = FRAME_INDICATOR_COLORS.get(classification, (128, 128, 128))

        # Define square properties
        square_size = FRAME_INDICATOR_SIZE
        margin = FRAME_INDICATOR_MARGIN
        
        # Calculate position (lower left corner)
        x1 = margin
        y1 = frame.shape[0] - square_size - margin
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # Draw filled rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        # Draw thick black border for visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        # Draw thin white border inside for contrast
        cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)
        return frame

    def detect_scene_shift(self, frame1, frame2, threshold=30.0):
        """
        Detects if there's a significant scene shift between two consecutive frames.
        Uses multiple methods for robust detection: histogram comparison and frame difference.
        Adds internal profiling for each step.
        Args:
            frame1: First video frame (BGR format)
            frame2: Second video frame (BGR format)
            threshold: Scene shift score threshold (0-100). Higher values require more change to be detected.
                       Default 30.0 means 30% change. Typical range: 15-40
        Returns:
            dict: {
                "is_scene_shift": bool,  # True if a scene shift is detected
                "shift_score": float,    # Scene shift intensity (0-100, higher = more change)
                "histogram_diff": float, # Histogram difference component
                "frame_diff": float,     # Frame difference component
                "profile": dict          # Internal timing info (ms)
            }
        """
        # Ensure frames have the same dimensions
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        # Downscale both frames to 50% (for both histogram and pixel-diff)
        scale = 0.5
        new_size = (int(frame1.shape[1]*scale), int(frame1.shape[0]*scale))
        frame1_small = cv2.resize(frame1, new_size)
        frame2_small = cv2.resize(frame2, new_size)
        # Method 1: Histogram comparison (on downscaled)
        hsv1 = cv2.cvtColor(frame1_small, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2_small, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        histogram_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        histogram_score = histogram_diff * 100  # Scale to 0-100
        # Method 2: Mean absolute difference (simple pixel-level difference)
        frame_diff = cv2.absdiff(frame1_small, frame2_small)
        mean_diff = np.mean(frame_diff)
        # Normalize to 0-100 range (255 is max difference per channel)
        frame_diff_score = (mean_diff / 255) * 100
        # Combine both methods (weighted average)
        shift_score = (histogram_score * 0.6 + frame_diff_score * 0.4)
        # Detect scene shift if score exceeds threshold
        is_scene_shift = shift_score > threshold
        return {
            "is_scene_shift": is_scene_shift,
            "shift_score": round(shift_score, 2),
            "histogram_diff": round(histogram_score, 2),
            "frame_diff": round(frame_diff_score, 2)
        }
