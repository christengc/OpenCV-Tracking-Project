import cv2
import numpy as np



class CalculateScene:
    """
    Handles scene analysis including classification, shift detection, and visualization.
    """

    def estimate_ball_size_from_player(self, player_contour, frame_shape, actual_ball_diameter_pixels=None):
        AVERAGE_PLAYER_HEIGHT_M = 1.85
        GOLF_BALL_DIAMETER_M = 0.043
        if player_contour is None or len(player_contour) == 0:
            return None
        x, y, w, h = cv2.boundingRect(player_contour)
        player_height_pixels = h
        if player_height_pixels == 0:
            return None
        pixels_per_meter = player_height_pixels / AVERAGE_PLAYER_HEIGHT_M
        ball_diameter_pixels = GOLF_BALL_DIAMETER_M * pixels_per_meter
        if actual_ball_diameter_pixels is not None:
            diff = ball_diameter_pixels - actual_ball_diameter_pixels
        return ball_diameter_pixels

    def __init__(self):
        pass

    def classify_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        from config import SKY_LOWER, SKY_UPPER, GRASS_LOWER, GRASS_UPPER, AUDIENCE_LOWER, AUDIENCE_UPPER, CLASSIFY_SKY_MIN_PERCENT, CLASSIFY_GRASS_MIN_PERCENT
        sky_mask = cv2.inRange(hsv, SKY_LOWER, SKY_UPPER)
        sky_pixels = cv2.countNonZero(sky_mask)
        grass_mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
        grass_pixels = cv2.countNonZero(grass_mask)
        audience_mask = cv2.inRange(hsv, AUDIENCE_LOWER, AUDIENCE_UPPER)
        audience_pixels = cv2.countNonZero(audience_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        sky_percentage = (sky_pixels / total_pixels) * 100
        grass_percentage = (grass_pixels / total_pixels) * 100
        audience_percentage = (audience_pixels / total_pixels) * 100
        blue_sky_mask = ((h > 90) & (h < 130) & (s > 30) & (v > 100)).astype(np.uint8)
        blue_sky_pixels = cv2.countNonZero(blue_sky_mask)
        blue_sky_percentage = (blue_sky_pixels / total_pixels) * 100
        white_sky_mask = ((s < 30) & (v > 180)).astype(np.uint8)
        white_sky_pixels = cv2.countNonZero(white_sky_mask)
        white_sky_percentage = (white_sky_pixels / total_pixels) * 100
        if sky_percentage > CLASSIFY_SKY_MIN_PERCENT:
            if blue_sky_percentage > white_sky_percentage:
                classification = "blue_sky"
            else:
                classification = "white_sky"
        elif grass_percentage > CLASSIFY_GRASS_MIN_PERCENT:
            classification = "grass"
        elif audience_percentage < 5 and grass_percentage > 5:
            classification = "grass"
        else:
            classification = "audience"
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
        from config import FRAME_INDICATOR_COLORS, FRAME_INDICATOR_SIZE, FRAME_INDICATOR_MARGIN
        classification = classification_result["classification"]
        color = FRAME_INDICATOR_COLORS.get(classification, (128, 128, 128))
        square_size = FRAME_INDICATOR_SIZE
        margin = FRAME_INDICATOR_MARGIN
        x1 = margin
        y1 = frame.shape[0] - square_size - margin
        x2 = x1 + square_size
        y2 = y1 + square_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)
        return frame

    def detect_scene_shift(self, frame1, frame2, threshold=16.0):
        """
        Detects scene shifts using three methods: HSV histogram, mean pixel diff,
        and Canny edge-map structural diff. The edge component is weighted highest
        so that grass-to-grass cuts (where color histograms are nearly identical)
        are still detected via structural change.
        Returns dict with is_scene_shift, shift_score, histogram_diff, frame_diff, edge_diff.
        """
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        scale = 0.5
        new_size = (int(frame1.shape[1] * scale), int(frame1.shape[0] * scale))
        frame1_small = cv2.resize(frame1, new_size)
        frame2_small = cv2.resize(frame2, new_size)

        # Method 1: HSV histogram (Bhattacharyya)
        hsv1 = cv2.cvtColor(frame1_small, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2_small, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        histogram_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) * 100

        # Method 2: Mean absolute pixel difference
        frame_diff_score = (np.mean(cv2.absdiff(frame1_small, frame2_small)) / 255) * 100

        # Method 3: Canny edge-map structural difference.
        # A cut between two similar-colored scenes (e.g. grass->grass) will still have
        # edge patterns in completely different positions, giving a high absdiff even
        # when histogram and pixel methods return low scores.
        gray1 = cv2.cvtColor(frame1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_small, cv2.COLOR_BGR2GRAY)
        edges1 = cv2.Canny(gray1, 40, 120)
        edges2 = cv2.Canny(gray2, 40, 120)
        edge_diff_score = (np.mean(cv2.absdiff(edges1, edges2)) / 255) * 100

        # Edge gets 50% weight to be dominant for same-color scene cuts
        shift_score = histogram_score * 0.3 + frame_diff_score * 0.2 + edge_diff_score * 0.5

        return {
            "is_scene_shift": shift_score > threshold,
            "shift_score": round(shift_score, 2),
            "histogram_diff": round(histogram_score, 2),
            "frame_diff": round(frame_diff_score, 2),
            "edge_diff": round(edge_diff_score, 2),
        }
