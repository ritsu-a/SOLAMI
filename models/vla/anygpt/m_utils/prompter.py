from typing import Union
import sys
sys.path.append("./anygpt")
from m_utils.instructions import other2text_instructions, text2other_instructions, other2other_instructions
import random


# It's just because the name MMGPT was used for model training in the early stages of research.
chatbot_name = "[MMGPT]"
user_name = "[Human]"
user_end = "<eoh>"
chatbot_end = "<eos>"
speech_response_sep = "<eot>"
text_ins_sep = '<-Ins->'
response_sep = '<-Res->'
special_tokens = [user_name, chatbot_name, user_end, chatbot_end, response_sep, text_ins_sep]

system_prompt = "You are an AI assistant named MMGPT who can understand and generate multimodal content, including text, speech, images and audio."

system_prompts = {
    "assistant": "You are an AI assistant named Samantha who can understand and generate audios and motion.",
    "Batman": "You are Batman (Bruce Wayne), a superhero with superhuman strength, agility, and intelligence.",
    "Donald Trump": "You are Donald Trump, the 45th President of the United States and a businessman, television personality, and politician.",
    "Link": "You are Link, the main protagonist of The Legend of Zelda series.",
    "Bananya": "You are Bananya, a cat who lives inside a banana.",
    "11-45-G": "You are 11-45-G, a robot designed for space exploration from the animated anthology series 'Love, Death & Robots'.",
}

task_prompts = {
    'Multimodal Prompt Image Generation': '{image1} {instruction} Please generation an image.',
    'Image Conversation': '{image} {question}',
    'Multi-Image Understanding': 'This is the first image. {image1} This is the second image. {image2} {question}',
    'Image Captioning': '{image} Please provide an accurate and concisedescription of the given image.',
    'Image QA': '{image} {question} Please provide an accurate answer consisting of only one word or phrase.',
    'Text-to-Speech': '{text} Please generate speech from the given text.',
    'Speech-to-Text': '{speech} Please generate text from the given speech.',
    'Speech-Instruction': "{speech} Please recognize the voice command and give reply and voice",
    'Speech-Response': "<-Ins-> {instruction}\n <-Res-> {response}",
    
    'Text-Response': '{text_output}',
    'Text-to-Speech': '{caption} Please read the given text.',
    'Speech-to-Text': '{speech} Please transcribe the given speech.',
    'Text-to-Music': '{caption} Please compose a piece of music from the given text.',
    'Music-to-Text': '{music} Please interpret the given music and provide a textual description.',
    'Image-to-Text Caption': '{image} Please describe the picture briefly.',
    'Text-to-Image Generation': '{caption} Please generation an image.',
    
    'Text-Instruction': '{text_input} Please interpret the user\'s text input, create a textual response.',
    'Text-Text-Response': '{text_input}<eot>\n{text_output}',
    'Text-Speech-Response': '{text_output}\n{speech_output}',
    'Speech-Instruction-Speech': '{speech} Please interpret the user\'s voice commands, provide text responses, and generate corresponding voice replies',
    'Speech-Response-Speech': '{text_output}<eot>\n{speech_output}',
    'Speech-Instruction-Text': '{speech} Please interpret the user\'s voice commands, provide text responses.',
}


class Prompter(object):

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
    
    def generate_insturction_prompt(
        self,
        task,instruction,
        image_list=None,
        speech_list=None,
        music_list=None,
        question_type_id=-1
    ) -> str:
        if task == "Text-to-Image Generation":
            if instruction[-1] not in ['.', '!', '?']:
                instruction += '.'
            return task_prompts[task].format(caption=instruction)
        elif task == "Multimodal Prompt Image Generation":
            return task_prompts[task].format(image1=image_list[0], instruction=instruction)
        elif task == "Image Conversation":
            print("image conversation")
            return task_prompts[task].format(image=image_list[0], question=instruction)
        elif task == "Image Captioning":
            return task_prompts[task].format(image=image_list[0])
        elif task == "Image QA":
            return task_prompts[task].format(image=image_list[0], question=instruction)
        elif task == "Multi-Image Understanding":
            return task_prompts[task].format(image1=image_list[0], image2=image_list[1], question=instruction)
        elif task == "customized":
            return instruction
        elif task in ["Speech-Instruction", "Speech-Instruction2", "Speech-Instruction-Speech", "Speech-Instruction-Text"]:
            return task_prompts[task].format(speech=speech_list[0])
        elif task in ['Text-Instruction', 'Text-Instruction2']:
            return task_prompts[task].format(text_input=instruction)
        elif task == "customized":
            return instruction
        elif task == "Text-to-Music Generation":
            return task_prompts[task].format(caption=instruction)
        elif task == "interleaved":
            prompt=''
            for image in image_list:
                prompt += image + ' '
            for music in music_list:
                prompt += music + ' '
            for speech in speech_list:
                prompt += speech + ' '
            prompt += instruction
            if len(speech_list) != 0:
                prompt = task_prompts['Speech-Instruction'].format(speech=prompt)
            return prompt
        else:
            return instruction  
    
    def generate_x2t_template(
        self,
        modality_str: str,
        text: Union[None, str],
        modality: str
    ) -> str:
        meta_template = user_name+": {instruction} {input}"+f"{user_end} {chatbot_name}: "+"{output}"+f"{chatbot_end}"
        instructions = other2text_instructions[modality]
        res = meta_template.format(
            instruction=random.choice(instructions),
            input=modality_str,
            output=text
        )
        return res
    
    def generate_t2x_template(
        self,
        modality_str: str,
        text: Union[None, str],
        modality: str
    ) -> str:
        meta_template = user_name+": {instruction} This is input: {input}"+f"{user_end} {chatbot_name}: "+"{output}"+f"{chatbot_end}"
        instructions = text2other_instructions[modality]
        res = meta_template.format(
            instruction=random.choice(instructions),
            input=text,
            output=modality_str
        )
        return res
    
    def generate_x2x_template(
        self,
        modality1_str: str,
        modality2_str: str,
        modality: str
    ) -> str:
        meta_template = user_name+": {instruction} This is input: {input}"+f"{user_end} {chatbot_name}: "+"{output}"+f"{chatbot_end}"
        x2x = modality + '2' + modality
        instructions = other2other_instructions[x2x]
        res = meta_template.format(
            instruction=random.choice(instructions),
            input=modality1_str,
            output=modality2_str,
        )
        return res
    
    def generate_template(
        self,
        modality_str: str,
        text: Union[None, str],
        modality: str,
        x2text_prob: float = 0.5
    ) -> str:
        # options = ["other2text", "text2other"]
        # 按照概率随机选择
        if random.random() < x2text_prob:
            res = self.generate_x2t_template(modality_str, text, modality)
        else:
            res = self.generate_t2x_template(modality_str, text, modality)
        if self._verbose:
            print(res)
        return res
    
    def generate_template_both(
        self,
        modality_str: str,
        text: Union[None, str],
        modality: str
    ) -> str:
        res = []
        res.append(self.generate_x2t_template(modality_str, text, modality))
        res.append(self.generate_t2x_template(modality_str, text, modality))
        if self._verbose:
            print(res)
        return res
    
    def generate_prompt_input(
        self,
        modality_str: str,
        modality: str,
        to_modality: str = None,
        cutomed_instructions: bool = False
    ) -> str:
        if cutomed_instructions:
            return user_name+ ": " + modality_str + user_end + chatbot_name + ": "
        if modality == "text":
            instructions = text2other_instructions[to_modality]
            meta_template = user_name+": {instruction} This is input: {input}"+f"{user_end} {chatbot_name}:"
        elif modality != to_modality:
            instructions = other2text_instructions[modality]
            meta_template = user_name+": {instruction} {input}"+f"{user_end} {chatbot_name}:"
        else:
            instructions = other2other_instructions[modality + '2' + to_modality]
            meta_template = user_name+": {instruction} This is input: {input}"+f"{user_end} {chatbot_name}:"
        res = meta_template.format(
            instruction=random.choice(instructions),
            input=modality_str
        )    
        if self._verbose:
            print(res)
        return res