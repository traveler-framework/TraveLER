"""
Utility functions for dealing with videos.
"""

import ast


class VideoObj():
    """
    A class to represent a video object. 
    Stores information about video, question, answer choices, memory bank, and more.
    Provides utility functions to navigate through the video.
    """
    
    def __init__(self, images, question, choices, vid_id, answer=None, query_type=None, start=None, end=None, key_region=None, reasons=None):
        
        self.images = images
        self.question = question
        
        choices_list = ast.literal_eval(choices)
        choice_dict = {index: choice for index, choice in enumerate(choices_list)}
        self.choices = choice_dict
        
        self.answer = answer
        self.vid_id = vid_id
        self.query_type = query_type
        self.length = len(images)
        self.fps = 30
        self.info = {}
        self.explanations = list()
        self.length_secs = self.length / self.fps
        self.start = start
        self.end = end
        self.key_region = key_region
        if reasons:
            reasons_list = ast.literal_eval(reasons)
            reasons_dict = {index: reason for index, reason in enumerate(reasons_list)}
            self.reasons = reasons_dict

    def __getitem__(self, index):
        """
        Get the frame at the given index.
        """
        return self.images[index]
    
    def __len__(self):
        """
        Get the length of the video in number of frames.
        """
        return self.length
    
    def get_second_from_frame(self, frame):
        """
        Convert a frame index to a second.
        """
        return frame / self.fps

    def get_frame_from_second(self, second):
        """
        Get a specific frame at a given second.
        """
        idx = int(second * self.fps)
        idx = min(idx, self.length - 1) # error check
        return [idx], self.images[idx]
     
    def convert_to_frame(self, second):
        """
        Convert a second to a frame index.
        """
        return int(second * self.fps)
        
    def select_seconds_range(self, start, end):
        """
        Get a range of frames from the video. 
        Enforces that the start and end are within the video length.
        """
        start_idx = min(0, int(start * self.fps))
        end_idx = min(int(end * self.fps), self.length)    
        return self.images[start_idx, end_idx]

    def get_block(self, second, length=1, method="center"):
        """
        Get a range of frames from the video.
            - length: length of the block in seconds
            - method: method to select the block. 
                "center" selects the block around the given second, 
                "start" selects the block starting at the given second, 
                "end" selects the block ending at the given second.
        """
        if method == "center":
            start = second - length // 2, 0
            end = second + length // 2
        elif method == "start":
            start = second
            end = second + length
        elif method == "end":
            start = second - length
            end = second
        return self.select_seconds_range(start, end)

    def move(self, current, direction, step_size=1):
        """
        Move forward or backward in the video by retrieving the corresponding block `step_size` seconds away.
        """
        assert direction in ["forward", "backward"]
        if direction == "forward":
            return self.get_block(current + step_size)
        elif direction == "backward":
            return self.get_block(current - step_size)
