"""
Answerer for Perception Test.
"""

import ast
import logging
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL
import yaml

import wandb
from modules import BaseModel
from vidobj import VideoObj

# ===== Config ===== #
config_path = os.path.join(os.environ["EXP_PATH"], "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# ===== Logging ===== #
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(os.path.join(os.getcwd(), config["results_dir"], config["experiment_name"], log_file.split(".")[0] + '_' + os.environ["OUTFILE_NAME"] + ".log"))
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Disable console logging
    logger.propagate = False
    return logger

prompt_log = setup_logger('prompt_logger', config["logger"]["prompts"])
output_log = setup_logger('output_logger', config["logger"]["outputs"])

def log_all(lst, content):
    for logger in lst:
        logger.info(content)

# ========================== Base answerer model ========================== #
class Answerer():
    """Main Answerer class. Contains all the modules and functions to run the Answerer."""
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, llm: BaseModel):
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm
        self.video = None
        self.max_tries = config["max_iters"]
        self.view_range = config["retriever"]["view_range"]

        print("===== Config =====")
        print(f"Max iters: {self.max_tries}")
        print(f"View range: {self.view_range}")
        print(f"Number of questions: {config['extractor']['num_qs']}")
        print("==================")

        self.planner = Planner(self)
        self.retriever = Retriever(self)
        self.extractor = Extractor(self)
        self.summarizer = Summarizer(self)
        self.evaluator = Evaluator(self)
        
    def construct_prompt(self, question: str, choices: List[str], video_info: Dict[str, List[str]], prompt_path: str) -> str:
        """Constructs prompt by inserting question, choices, and video info into the prompt template."""
        with open(os.path.join(os.environ["EXP_PATH"], prompt_path)) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_CHOICES_HERE", str(choices)).replace("INSERT_INFO_HERE", str(video_info))
        return prompt
    
    def merge_dicts(self, dict1: Dict[str, List[str]], dict2: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merges two dictionaries and returns the merged dictionary."""
        merged_dict = {}
        # Merge dict1 into the merged_dict
        for key, value in dict1.items():
            if key in merged_dict:
                # If the key exists, extend the existing list
                merged_dict[key].extend(value)
            else:
                # If the key doesn't exist, add the key and its list
                merged_dict[key] = value

        # Merge dict2 into the merged_dict
        for key, value in dict2.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
        return merged_dict

    def construct_timestamp(self, start_sec: float, end_sec: Optional[float] = None) -> str:
        """Constructs the timestamp string from the start and end seconds."""
        def format_seconds(seconds):
            minutes, sec = divmod(seconds, 60)
            # Format the string to include leading zeros and limit the fractional part to two digits
            formatted_time = f"{int(minutes):02d}:{int(sec):02d}.{int((sec - int(sec)) * 100):02d}"
            return formatted_time
        start_sec_string = format_seconds(start_sec)
        if end_sec is None:
            end_sec = self.video.length_secs
        end_sec_string = format_seconds(end_sec)
        return start_sec_string + "/" + end_sec_string

    def construct_info(self, start_sec: float, end_sec: float, answer: str, question: str = None, type: str = "caption") -> Tuple[str, str]:
        """Constructs the key and info pair for the video info dictionary.

        Args:
            start_sec: the start time of the keyframe in seconds.
            end_sec: the end time of the keyframe in seconds.
            answer: the answer to the question, or caption
            question: the question, if it is a QA pair.
            type: the type of the info. can be "caption" or "qa".
        """
        end_sec = self.video.get_second_from_frame(len(self.video))
        key = self.construct_timestamp(start_sec, end_sec)
        if type == "caption": # caption case
            return key, answer
        else: # qa case
            qa_pair = f"Question - {question} Answer - {answer}"
            return key, qa_pair
    
    def frange(self, start, stop, step: float=1.0) -> List[float]:
        """Function to generate a range of floats."""
        lst = []
        i = start
        while i < stop + step:
            lst.append(i)
            i += step
        return lst

    def init_update(self) -> Tuple[List[float], List[Tuple[int, PIL.Image.Image]], List[str]]:
        """Function that initializes the memory bank."""
        ts = time.time()
        
        gotos = [c * self.video.length_secs for c in np.linspace(0, 1, 5)]
        frames = [self.video.get_frame_from_second(i) for i in gotos]

        key = self.construct_timestamp(0.5*self.video.length_secs)

        output = self.extractor.forward(images=frames, plan=None, start_secs=gotos, qa=False)
        self.retriever.curr = key # updates the retriever's current timestamp

        te = time.time()
        if config["wandb"]:
            wandb.log({"init_time": te - ts}, commit=False)

        try:
            init_caption = output
        except Exception:
            print("ERROR in Answerer `init_update`.")
            print(output)
            print(traceback.format_exc())
        
        self.video.info = init_caption

        output_log.info(f"INIT GOTO: {gotos}, INIT CAPTION: {init_caption}")
        return gotos, frames, init_caption
    
    def reset_history(self) -> None:
        """Function to reset the states of the Planner."""
        self.planner.plan = None

    def forward(self, video: VideoObj) -> int:
        """Forward function for Answerer."""

        self.video = video
        self.planner.explanation = None
        self.retriever.curr = None
        start_secs, start_images, init_caption = self.init_update()

        for i in range(self.max_tries): 

            if i != 0: # don't reset history on first try because we need the inital keyframes
                self.reset_history()

            log_all([prompt_log, output_log], f"TRY: {i}/{self.max_tries}")
            
            ts_answerer = time.time()
            
            # ----- Planner -----
            ts = time.time()
            if i == 0: # first try planner feeds in SigLIP Keyframe
                plan = self.planner.forward(self.video, init_caption=init_caption)
            else:
                plan = self.planner.forward(self.video, init_caption=None)
            te = time.time()
            
            if config["wandb"]:
                wandb.log({"planner_time": te - ts}, commit=False)
            output_log.info(f'PLAN: {plan}')
            
            # ----- Retriever -----
            ts = time.time()
            if i == 0: # first try retriever bypass since we use the memory initialization
                new_tstmps, frames = start_secs, start_images
            else:
                new_tstmps, frames = self.retriever.forward(self.video, plan)
            te = time.time()

            if config["wandb"]:
                wandb.log({"retriever_time": te - ts}, commit=False)
            output_log.info(f"RETRIEVER: {new_tstmps}")
            
            # ----- Extractor -----
            ts = time.time()
            output = self.extractor.forward(frames, plan, new_tstmps, caption=True, qa=True)
            te = time.time()

            if config["wandb"]:
                wandb.log({"extractor_time": te - ts}, commit=False)
            output_log.info(f"EXTRACTOR: {output}")

            # ----- Summarizer -----
            ts = time.time()
            summary = self.summarizer.forward(output)
            output_log.info(f"SUMMARY: {summary}")
            self.video.info = self.merge_dicts(self.video.info, summary)
            self.video.info = dict(sorted(self.video.info.items(), key=lambda x: x[0]))
            output_log.info(f"MEMORY: {self.video.info}")
            te = time.time()

            if config["wandb"]:
                wandb.log({"summarizer_time": te - ts}, commit=False)
            
            # ----- Evaluator ----- 
            ts = time.time()
            final_choice = self.evaluator.forward(self.video.question, self.video.choices, self.video, plan)
            te = time.time()

            if config["wandb"]:
                wandb.log({"evaluator_time": te - ts}, commit=False)
            output_log.info(f"FINAL CHOICE: {final_choice}")
            
            te_answerer = time.time()
            if config["wandb"]:
                wandb.log({"answerer_time": te_answerer - ts_answerer})

            # free images in videoobj
            if final_choice is not None:
                for image in self.video.images:
                    del image
                self.video.images.clear()
                self.video = None
                return final_choice
        return self.evaluator.forward(self.video.question, self.video.choices, self.video, plan, final_choice=True)

# ========================== Base planner model ========================== #
    
class Planner():
    def __init__(self, answerer):
        self.answerer = answerer
        self.plan = None
        self.explanation = None

    def create_plan(self, video: VideoObj, init_caption: Optional[str] = None) -> str:
        first_prompt_path = config["planner"]["planner_first_prompt"]
        prompt_path = config["planner"]["planner_prompt"]
        if init_caption is not None: # other than the first time, init_caption should be None, and we use the past info instead
            prompt = self.answerer.construct_prompt(video.question, str(video.choices), init_caption, first_prompt_path) \
                .replace("INSERT_LENGTH_HERE", str(video.length_secs))
        else:
            prompt = self.answerer.construct_prompt(video.question, str(video.choices), video.info, prompt_path) \
                .replace("INSERT_LENGTH_HERE", str(video.length_secs)) \
                .replace("INSERT_EXPLANATION_HERE", str(self.explanation))
        
        prompt_log.info(f"PLANNER PROMPT: {prompt}")

        output = self.answerer.llm.forward(prompt)
        return output

    def forward(self, video: VideoObj, init_caption: Optional[str] = None) -> str:
        """Forward method for Planner."""
        output = self.create_plan(video, init_caption)
        return output

# ========================== Base retriever model ========================== #
    
class Retriever():
    def __init__(self, answerer):
        self.answerer = answerer
        self.curr = None

    def select_frame(self, video: VideoObj, plan: str) -> str:
        prompt_path = config["retriever"]["retriever_prompt"]
        prompt = self.answerer.construct_prompt(video.question, video.choices, video.info, prompt_path) \
            .replace("INSERT_PLAN_HERE", plan) \
            .replace("INSERT_LENGTH_HERE", str(self.answerer.video.length_secs)) \
            .replace("INSERT_CURR_HERE", str(self.curr))
        
        prompt_log.info(f"RETRIEVER PROMPT: {prompt}")

        output = self.answerer.llm.forward(prompt)
        return output
    
    def parse_answer(self, answer: str) -> Tuple[List[float], List[Tuple[int, PIL.Image.Image]]]:
        try:
            goto = float(answer)
            goto = min(goto, self.answerer.video.length_secs) # to check for out of bounds

            min_goto = max(goto - self.answerer.view_range, 0) # to check for out of bounds
            max_goto = min(goto + self.answerer.view_range, self.answerer.video.length_secs) # to check for out of bounds
            
            gotos = self.answerer.frange(min_goto, max_goto)
            frames = [self.answerer.video.get_frame_from_second(i) for i in gotos]

            curr_string = self.answerer.construct_timestamp(goto)
            self.curr = curr_string
            return gotos, frames
        except Exception:
            print("ERROR in Retriever `parse_answer`.")
            print(answer)
            print(traceback.format_exc())
    
    def forward(self, video: VideoObj, plan: str) -> Tuple[List[float], List[Tuple[int, PIL.Image.Image]]]:
        """Forward function for Retriever."""
        output = self.select_frame(video, plan)
        gotos, frames = self.parse_answer(output)
        return gotos, frames
    
# ========================== Base extractor model ========================== #
    
class Extractor(Answerer):
    def __init__(self, answerer):
        self.answerer = answerer
        self.num_qs = config["extractor"]["num_qs"]

    def preprocess_caption(self) -> str:
        return "Describe the scene in less than 2 sentences. Answer:"
    
    def preprocess_question(self, question: str) -> str:
        question_header = "If there are factual errors in the questions, point it out; if not, proceed answering the question. Explain your answer, but keep it brief. Answer:"
        return question_header + question

    def get_questions(self, video: VideoObj, plan: str, start_secs: List[float], captions: List[str]) -> List[str]:
        prompts = []
        prompt_path = config["extractor"]["extractor_prompt"]
        
        for start_sec, caption in zip(start_secs, captions):
            if video.info == {}:
                prompt = self.answerer.construct_prompt(video.question, video.choices, caption, prompt_path) \
                    .replace("INSERT_PLAN_HERE", plan) \
                    .replace("INSERT_LENGTH_HERE", str(video.length_secs)) \
                    .replace("INSERT_CURRENT_TIME_HERE", str(start_sec)) \
                    .replace("INSERT_FRAME_CAPTION_HERE", caption) \
                    .replace("INSERT_NUM_QUESTIONS", str(self.num_qs))
            else:
                prompt = self.answerer.construct_prompt(video.question, video.choices, video.info, prompt_path) \
                    .replace("INSERT_PLAN_HERE", plan) \
                    .replace("INSERT_LENGTH_HERE", str(video.length_secs)) \
                    .replace("INSERT_CURRENT_TIME_HERE", str(start_sec)) \
                    .replace("INSERT_FRAME_CAPTION_HERE", caption) \
                    .replace("INSERT_NUM_QUESTIONS", str(self.num_qs))
            
            prompt_log.info(f"EXTRACTOR PROMPT: {prompt}")
            prompts.append(prompt)
        
        output = self.answerer.llm.forward(prompts)
        return output
    
    def parse_answers(self, answers: List[str]) -> List[List[str]]:
        parsed_answers = []
        for answer in answers:
            try:
                parsed_answer = ast.literal_eval(answer) # supposed to be list
                parsed_answers.append(parsed_answer)
            except Exception:
                print("ERROR in Extractor `parse_answers`.")
                print(answer)
                print(traceback.format_exc())
        return parsed_answers

    def forward(self, images: List[PIL.Image.Image], plan: str, start_secs: List[float], caption: bool=True, qa: bool=True) -> Dict[str, List[str]]:
        """Forward function for Extractor."""
        output = {}
        
        # number of captions / questions per image
        counts = []
        # all images in this batch
        batch_images = []
        # all questions to ask in this batch
        batch_questions = []
        raw_questions = []
       
        caption_prompts = [] 
        caption_images = [] 

        ts_create = time.time()

        # caption all images first
        if caption:
            for i, start_sec in enumerate(start_secs):
                caption_prompts.append(self.preprocess_caption())
                caption_images.append(images[i][1])
            
            ts_caption = time.time()
            captions = self.answerer.vqa_model.forward(caption_images, caption_prompts)
            te_caption = time.time()
            
            if config["wandb"]:
                wandb.log({"caption_time": te_caption - ts_caption}, commit=False)
        
        if qa:
            for _ in range(config["max_retries"]):
                try:
                    questions_unparsed = self.get_questions(self.answerer.video, plan, start_secs, captions)
                    questions_list = self.parse_answers(questions_unparsed) 
                    break
                except Exception:
                    print("ERROR in Extractor `forward`.")
                    print(questions_unparsed)
                    print(traceback.format_exc())
                    print("=== RETRYING EXTRACTOR... ===")
                    continue
            
            for i, questions in enumerate(questions_list):
                output_log.info(f"QUESTIONS: {questions}")
                prepared_questions = [self.preprocess_question(question) for question in questions]
                raw_questions.extend(questions)
                batch_questions.extend(prepared_questions)
                
                count = len(questions)
                counts.append(count)
                batch_images.extend([images[i][1] for _ in range(count)])
                
            # batch inference
            ts_vqa = time.time()
            answers = self.answerer.vqa_model.forward(batch_images, batch_questions)
            te_vqa = time.time()
            
            if config["wandb"]:
                wandb.log({"qa_time": te_vqa - ts_vqa}, commit=False)

        te_create = time.time()
        if config["wandb"]:
            wandb.log({"create_time": te_create - ts_create}, commit=False)
        
        # weird looping bc each image may have arbitrary number of questions (count)
        # push the flattened answers back into the memory dict
        ts_reconstruct = time.time()
        i = 0
        for j in range(len(captions)):
            
            results = []
            
            if caption:
                count = 1
                key, text = self.answerer.construct_info(start_secs[j], end_sec=self.answerer.video.length_secs, answer=captions[j])
                results.append(text)
            if qa:
                count = counts[j]
                for k in range(count):
                    key, qa_pair = self.answerer.construct_info(start_secs[j], end_sec=self.answerer.video.length_secs, answer=answers[i + k], question=raw_questions[i + k], type="qa")
                    results.append(qa_pair)
            
            output[key] = results
            i += count
            
        te_reconstruct = time.time()
        if config["wandb"]:
            wandb.log({"reconstruction_time": te_reconstruct - ts_reconstruct}, commit=False)
            
        counts = []
        
        batch_images = None
        caption_images = None
        images = None
        
        return output
        
# ========================== Base summarizer model ========================== #

class Summarizer(Answerer):
    def __init__(self, answerer):
        self.answerer = answerer
        self.prompt = os.path.join(os.environ["EXP_PATH"], config["summarizer"]["summarizer_prompt"])

    def get_summary(self, info: Dict[str, List[str]], video: VideoObj) -> str:
        with open(self.prompt) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_INFO_HERE", str(info)).replace("INSERT_QUESTION_HERE", str(video.question)).replace("INSERT_CHOICES_HERE", str(video.choices))
        return prompt
    
    def parse_output(self, output: str) -> Dict[str, List[str]]:
        return ast.literal_eval(output)

    def forward(self, info: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Foward function for Summarizer."""
        prompt = self.get_summary(info, self.answerer.video)
        for _ in range(config["max_retries"]):
            try:
                raw_output = self.answerer.llm.forward(prompt)
                output = self.parse_output(raw_output)
                break
            except Exception:
                print("ERROR in Summarizer `forward`.")
                print(raw_output)
                print(traceback.format_exc())
                print("=== RETRYING SUMMARIZER... ===")
                continue
        return output

# ========================== Base evaluator model ========================== #
    
class Evaluator(Answerer):
    def __init__(self, answerer):
        self.answerer = answerer

    def evaluate_info(self, question: str, choices: List[str], video: VideoObj, plan: str) -> str:
        prompt_path = config["evaluator"]["evaluator_prompt"]
        choices_str = "{"
        for i in range(len(choices)):
            choices_str += f"{i}: {choices[i]}"
            if i < len(choices) - 1:
                choices_str += ", "
        choices_str += "}"
        prompt =  self.answerer.construct_prompt(question, choices_str, video.info, prompt_path) \
            .replace("INSERT_LENGTH", str(video.length_secs)) \
            .replace("INSERT_PLAN_HERE", plan)

        prompt_log.info(f"EVALUATOR PROMPT: {prompt}")

        output = self.answerer.llm.forward(prompt)
        return output

    def final_select(self, question: str, choices: List[str], video_info: Dict[str, List[str]]) -> str:
        prompt_path = config["evaluator"]["final_select"]
        prompt =  self.answerer.construct_prompt(question, choices, video_info, prompt_path).replace("INSERT_LENGTH", str(self.answerer.video.length_secs))

        prompt_log.info(f"FINAL_SELECT PROMPT: {prompt}")

        output = self.answerer.llm.forward(prompt)
        return output

    def parse_output(self, answer: str) -> Tuple[int, str]:
        try:
            final_ans = answer.split("Final Answer:")[1].strip().split(':')[0]
            explanation = answer.split("Final Answer:")[0].strip()
            answer = ast.literal_eval(final_ans)
            if answer not in [None, "None"]:
                answer = int(answer)
            return answer, explanation
        except Exception:
            print("ERROR in Evaluator `parse_output`.")
            print(answer)
            print(traceback.format_exc())
    
    def forward(self, question: str, choices: List[str], video: VideoObj, plan: str, final_choice: bool = False):
        """Forward function for Evaluator."""
        for _ in range(config["max_retries"]):
            try:
                if not final_choice:
                    output = self.evaluate_info(question, choices, video, plan)
                else:
                    output = self.final_select(question, choices, video.info)

                final_output, explanation = self.parse_output(output)
                self.answerer.planner.explanation = explanation
            except Exception:
                print("ERROR in Evaluator `forward`.")
                print(output)
                print(traceback.format_exc())
                print("=== RETRYING EVALUATOR... ===")
                continue
            
            if final_choice:
                for image in self.answerer.video.images:
                    del image
                self.answerer.video.images.clear()
                for info in self.answerer.video.info:
                    del info
                self.answerer.video.info.clear()
                self.answerer.video = None
            
            return final_output
