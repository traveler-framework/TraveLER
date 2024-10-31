"""
Datasets for TraveLER.
"""

import os
from typing import List

import cv2
import pandas as pd
import torchvision.transforms as T
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset

from vidobj import VideoObj


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        query_file: str,
        start_sample: int = 0,
        max_samples: int = 100,
        evaluation: bool = False,
    ):
        """
        Base class for our Datasets.

        Args:
            data_path (str): file path of the videos
            query_file (str): file path of the query file
            image_transforms: image transforms to apply to the frames
            start_sample (int): start index of the query file
            max_samples (int): maximum number of samples to read from query file
            evaluation (bool): if true, no correct answer known (test set)

        Returns:
            None
        """
        self.path = data_path
        self.query_file = query_file
        self.start_sample = start_sample
        self.max_samples = max_samples
        self.evaluation = evaluation

        with open(self.query_file) as f:
            self.df = pd.read_csv(f, index_col=None, keep_default_na=False)

        if self.max_samples is not None:
            self.df = self.df[start_sample : start_sample + self.max_samples]

        self.length = len(self.df)
        self.segment = False # only true for STAR
        self.reasons = False # only true for CausalVidQA

    def __getitem__(self, index):
        row_dict = self.df.iloc[index].to_dict()
        sample_path = self.get_sample_path(index)

        # get segment of video instead of full video
        if self.segment:
            start = self.df.iloc[index]["start"]
            end = self.df.iloc[index]["end"]
            video = self.get_video(
                sample_path, start=start, end=end
            )
        else:
        # get full video
            video = self.get_video(sample_path)
        row_dict["video"] = video
        video = None
        row_dict["index"] = index
        return row_dict

    def __len__(self):
        return self.length

    def get_sample_path(self, index: int) -> str:
        """
        Get file path for a sample video.
        
        Args:
            index (int): index of query file to read
        
        Returns:
            str: the file path for sample video
        """
        return os.path.join(self.path, self.df.iloc[index]["video_name"])

    def get_video(
        self,
        video_path,
        fps=30,
        sample_freq=None,
        start=0,
        end=0,
    ) -> List[ImageFile]:
        """
        Get a list of PIL images or transformed frames from file path of video.
        We interpolate the input video such that we get 10fps by default.
        
        Args:
            video_path (str): file path to the video
            fps: target fps to sample the video
            sample_freq: frequency of sampling the video frames. 
                sample_freq=10 means every 10th frame is sampled. 
                if None, all frames are sampled
            
        Returns:
            list: A list containing
                - PIL.Image: the frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video file.")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ratio = original_fps / fps

        if self.segment:
            start_frame = int(start * original_fps)
            end_frame = min(int(end * original_fps), vlen)
        else:
            start_frame = 0
            end_frame = vlen

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        transformed_video = []
        frame_id = start_frame # Start from the first frame of interest
        while frame_id < end_frame:
            ret = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id)) # Jump to the frame
            if not ret:
                print(
                    f"Skipping invalid frame at position {frame_id} for video {video_path}."
                )
                frame_id += ratio if sample_freq is None else sample_freq
                continue
            ret, frame = cap.read() # Read the frame
            if not ret:
                print(
                    f"Failed to read frame at position {frame_id} for video {video_path}."
                )
                break
            if sample_freq is None or (frame_id - start_frame) % sample_freq == 0:
                # Convert to PIL Image without any other transform
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                transformed_frame = T.ToPILImage()(frame)
                transformed_video.append(transformed_frame)
            frame_id += (
                ratio if sample_freq is None else sample_freq
            )  # Adjust frame_id increment based on fps and sample_freq

        cap.release()
        return transformed_video

    def construct_video(self, item) -> VideoObj:
        """
        Constructs a video object from a item dictionary
        representing a row in the query file.
        """
        
        if self.segment and "start" in item and "end" in item:
            # for star
            video_obj = VideoObj(
                images=item["video"],
                question=item["query"],
                choices=item["possible_answers"],
                vid_id=item["video_name"],
                answer=item["answer"],
                query_type=item["query_type"],
                start=item["start"],
                end=item["end"],
            )
        elif self.reasons:
            # for causalvidqa
            video_obj = VideoObj(
                images=item["video"],
                question=item["query"],
                choices=item["possible_answers"],
                vid_id=item["video_name"],
                query_type=item["query_type"],
                reasons=item["possible_reasons"],
            )
        elif self.evaluation:
            video_obj = VideoObj(
                images=item["video"],
                question=item["query"],
                choices=item["possible_answers"],
                vid_id=item["video_name"],
            )
        elif "key_region" in item:
            video_obj = VideoObj(
                images=item["video"],
                question=item["query"],
                choices=item["possible_answers"],
                vid_id=item["video_name"],
                answer=item["answer"],
                query_type=item["query_type"],
                key_region=item["key_region"],
            )
        else:
            video_obj = VideoObj(
                images=item["video"],
                question=item["query"],
                choices=item["possible_answers"],
                vid_id=item["video_name"],
                answer=item["answer"],
                query_type=item["query_type"],
            )
        return video_obj


class NextQADataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        query_file: str,
        start_sample: int = 0,
        max_samples: int = 5000,
        evaluation: bool = False,
    ):
        """
        Dataset for NExT-QA.
        """
        super().__init__(
            data_path,
            query_file,
            start_sample,
            max_samples,
            evaluation,
        )


class PerceptionTestDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        query_file: str,
        start_sample: int = 0,
        max_samples: int = 20000,
        evaluation: bool = True,
    ):
        """
        Dataset for Perception Test.
        """
        super().__init__(
            data_path,
            query_file,
            start_sample,
            max_samples,
            evaluation,
        )


class STARDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        query_file: str,
        start_sample: int = 0,
        max_samples: int = None,
        evaluation: bool = False,
    ):
        """
        Dataset for STAR.
        """
        super().__init__(
            data_path,
            query_file,
            start_sample,
            max_samples,
            evaluation,
        )
        self.segment = True


class EgoSchemaDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        query_file: str,
        start_sample: int = 0,
        max_samples: int = None,
        evaluation: bool = True,
    ):
        """
        Dataset for EgoSchema.
        """
        super().__init__(
            data_path,
            query_file,
            start_sample,
            max_samples,
            evaluation,
        )

    def get_video(self, video_path, fps=30, sample_freq=None):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video file.")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ratio = original_fps / fps

        start_frame = 0
        end_frame = vlen
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_ids = []
        frame_id = start_frame  # Start from the first frame of interest
        while frame_id < end_frame:
            frame_ids.append(frame_id)
            frame_id += ratio if sample_freq is None else sample_freq  # Adjust frame_id increment based on fps and sample_freq
        
        cap.release()
        return frame_ids


class CausalVidQADataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        query_file: str,
        start_sample: int = 0,
        max_samples: int = None,
        evaluation: bool = True,
    ):
        """
        Dataset for Causal-VidQA.
        """
        super().__init__(
            data_path,
            query_file,
            start_sample,
            max_samples,
            evaluation,
        )
        self.reasons = True
