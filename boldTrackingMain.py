import sys

from video_processor import VideoProcessor
from config import DEFAULT_VIDEO_PATH


def main():
    """Entry point for the tracker application.

    The first command‑line argument is treated as the path to the
    video file. When no argument is given DEFAULT_VIDEO_PATH from config
    is used.
    """
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    processor = VideoProcessor(video_path)
    processor.run()


if __name__ == "__main__":
    main()
