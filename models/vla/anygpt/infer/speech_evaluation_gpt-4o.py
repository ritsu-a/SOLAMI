import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub"
import numpy as np
import json
import argparse
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import debugpy

def initialize_debugpy():
    print("Debugpy is listening on port 15696")
    debugpy.listen(("0.0.0.0", 15696))
    debugpy.wait_for_client()
    

# initialize_debugpy()

AGENT_SETTINGS = {
    "User": "User is a human who is interacting with the character. User can ask questions, provide information, and engage in conversation with the character.",
    "Samantha": "Samantha is a 3D virtual companion, she has all the capabilities of a normal AI assistant. In addition, she can understand the human\'s body language, interact with human in real time, and perform sports, dance, and other skills with its body.",
    "Batman": "Batman (Bruce Wayne) is a superhero with superhuman strength, agility, and intelligence. He is a skilled martial artist, detective, and inventor. He has a strong sense of justice and is dedicated to protecting Gotham City from crime and corruption.",
    "Trump": "Donald Trump, the 45th President of the United States, is a businessman, television personality, and politician. He is known for his controversial statements and policies.",
    "Link": "Link, the main protagonist of The Legend of Zelda series, is a courageous hero who fights to save the kingdom of Hyrule from the evil sorcerer Ganon. He is skilled with a sword and shield and has the ability to solve puzzles and navigate dungeons.",
    "Banaya": "Bananya, a cat who lives inside a banana, is a curious and playful character who loves to explore the world around him. She has a childlike innocence and a sense of wonder about the world. Sometime she can be a little mischievous and crybaby.",
    "11-45-G": "11-45-G, a robot designed for space exploration from the animated anthology series 'Love, Death & Robots', is programmed to assist humans in their missions and is capable of performing complex tasks in extreme environments.",
    }

TOKENS_BLOG = {}

def ChatGPT_request_messages(messages, 
                             model="gpt-4o",
                             api_key='',
                             base_url='https://api.openai.com/v1',
                             temperature=1.,
                             presence_penalty=0.0,
                             client='openai',
                             tokens_blog={}, 
                             json_format=False,
                             **kwargs): 
    try: 
        if client == 'openai':
            client = OpenAI(api_key=api_key, base_url=base_url)
        elif client == 'azure':
            if api_key == '':
                api_key = os.environ["AZURE_OPENAI_API_KEY"]
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
            else:
                azure_endpoint = base_url
            client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version="2024-05-01-preview")
        else:
            raise ValueError("Invalid client")
        # client = OpenAI(api_key=api_key, base_url=base_url)
        response_format = {"type": "json_object" if json_format else "text"}
        if type(messages) is not list:
            messages = [{"role": "user", "content": messages}]
            completion = client.chat.completions.create(
            model=model, 
            messages=messages,
            response_format=response_format,
            presence_penalty=presence_penalty,
            temperature=temperature,
            )
        else:
            completion = client.chat.completions.create(
            model=model, 
            messages=messages,
            response_format=response_format,
            presence_penalty=presence_penalty,
            temperature=temperature,
            )
        
        if model not in tokens_blog.keys():
            tokens_blog[model] = {'completion': 0, 'prompt': 0}
        tokens_blog[model]['completion'] += completion.usage.completion_tokens
        tokens_blog[model]['prompt'] += completion.usage.prompt_tokens
        if json_format:
            return json.loads(completion.choices[0].message.content)
        else:
            return completion.choices[0].message.content
    except: 
        print ("ChatGPT RETURN ERROR")
        return False



def calculate_api_cost(tokens_blog={}):
    cost_dict = {
        'gpt-4o': {
            'prompt': 2.5,
            'completion': 10.,
        },
        'gpt-4o-2024-05-13': {
            'prompt': 5.,
            'completion': 15.,
        },
        'gpt-3.5-turbo-0125': {
            'prompt': 0.5,
            'completion': 1.5,
        },
        'gpt-3.5-turbo-instruct': {
            'prompt': 1.5,
            'completion': 2.0,
        },
        'text-embedding-ada-002': {
            'prompt': 0.1,
        },
        'text-embedding-3-small': {
            'prompt': 0.02,
        },
        'text-embedding-3-large': {
            'prompt': 0.13,
        }
    }
    
    Cost_all = 0
    Cost = {}
    for model, tokens in tokens_blog.items():
        Cost[model] = {}
        if model in cost_dict.keys():
            for token_type, token_num in tokens_blog[model].items():
                if token_type in cost_dict[model].keys():
                    Cost[model][token_type] = token_num * cost_dict[model][token_type] / 1000000
                    Cost_all += Cost[model][token_type]
    
    return Cost_all, Cost



#### load data files
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-0_evaluation")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    conv_ids = os.listdir(data_dir)
    ### pop non directory files
    conv_ids = [conv_id for conv_id in conv_ids if os.path.isdir(os.path.join(data_dir, conv_id))]
    
    results_save_path = os.path.join(data_dir, "gpt-4o_results.json")
    if os.path.exists(results_save_path):
        with open(results_save_path, "r") as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
    
    for conv_id in tqdm(conv_ids):
        conv_dir = os.path.join(data_dir, conv_id)
        
        if conv_id in eval_results.keys():
            continue

        ### read record
        with open(os.path.join(conv_dir, "record.json"), "r") as f:
            record = json.load(f)

        ### get gt_conversation and pred_conversation
        
        round_ids = list(record['chat'].keys())
        round_ids = [int(round_id) for round_id in round_ids]
        
        gt_convs = []
        pred_convs = []
    
        
        for r_id in round_ids:
            round_id = str(r_id)
            
            if 'pred' not in record['chat'][round_id]:
                gt_convs.append([record['chat'][round_id]['gt']['role'], record['chat'][round_id]['gt']['speech_text']])
                pred_convs.append([record['chat'][round_id]['gt']['role'], record['chat'][round_id]['gt']['speech_text']])
            else:
                gt_convs.append([record['chat'][round_id]['gt']['role'], record['chat'][round_id]['gt']['speech_text']])
                
                role_user = "User"
                role_agent = record['chat']["1"]['gt']['role']
                user_des = AGENT_SETTINGS[role_user]
                agent_des = AGENT_SETTINGS[role_agent]
                topic_des = record['topic']
                ### get background
                base_prompt = """
                Assume you are a script evaluator assessing the quality of the next line in an AI-generated script. 
                Please rate the line in two dimensions:
                Relevance to Topic and Context (marked as 'Relevance'): Rate from 1 to 5.
                Consistency with Character Traits (marked as 'Consistency'): Rate from 1 to 5.
                The value of 'Relevance' indicates how well the line fits the topic and context of the conversation.
                The value of 'Consistency' indicates how well the line aligns with the character traits and behavior of the speaker.
                ####
                """
                background_prompt = f"""
                The conversation is between {role_user} and {role_agent}.
                {user_des}
                Character Description: {agent_des}
                They are discussing about {topic_des}.
                ####\n\n
                """
                ref_prompt = "Reference scripts are as follow: \n"
                for i, gt_conv in enumerate(gt_convs):
                    ref_prompt += f"{gt_conv[0]}: {gt_conv[1]}\n"
                ref_prompt += "####\n\n"
                
                generated_prompt = "Here is the AI generated script: \n"
                generated_prompt += "Previous lines: \n"
                for i, pred_conv in enumerate(pred_convs):
                    generated_prompt += f"{pred_conv[0]}: {pred_conv[1]}\n"
                generated_prompt += "\nNow current line is: \n"
                generated_prompt += f"{record['chat'][round_id]['gt']['role']}: {record['chat'][round_id]['pred']['speech_text']}\n"
                generated_prompt += "Please evaluate the current line in two dimensions: Relevance and Consistency \n\n"
                
                example_prompt = """
                The response format is shown using an example as follow:
                {"Relevance": 4, "Consistency": 3}
                
                Be sure to output your evaluations according to the json format of the examples.
                So your response is:
                """
                prompt = base_prompt + background_prompt + ref_prompt + generated_prompt + example_prompt
                # print(prompt)
                response = False
                for i in range(3):
                    response = ChatGPT_request_messages(prompt, 
                                                        tokens_blog=TOKENS_BLOG, 
                                                        model="gpt-4o", 
                                                        temperature=1., 
                                                        presence_penalty=0.0, 
                                                        client='azure', 
                                                        json_format=True,
                                                        api_key="$YOUR_API_KEY",
                                                        base_url="$YOUR_ENDPOINT",
                                                        )
                    if response:
                        if "Relevance" in response.keys() and "Consistency" in response.keys():
                            if response["Relevance"] in range(1, 6) and response["Consistency"] in range(1, 6):
                                break
                
                if response is False:
                    print(f"Error in response: round {round_id} in conversation {conv_id}")
                else:
                    if conv_id not in eval_results.keys():
                        eval_results[conv_id] = {}
                    if round_id not in eval_results[conv_id].keys():
                        eval_results[conv_id][round_id] = [response]
                    else:
                        eval_results[conv_id][round_id].append(response)
                                 
                pred_convs.append([record['chat'][round_id]['gt']['role'], record['chat'][round_id]['pred']['speech_text']])
        ### write eval_results
        with open(results_save_path, "w") as f:
            json.dump(eval_results, f, indent=4)

    cost_all, costs = calculate_api_cost(TOKENS_BLOG)
    print('Total cost ($): {}'.format(str(cost_all)))
    print('Detail costs ($): ')
    print(json.dumps(costs, indent=4))

    ### calculate agerage score mean of relevance and consistency
    count = 0
    relevance_sum = 0
    consistency_sum = 0
    for conv_id, conv in eval_results.items():
        for round_id, round in conv.items():
            for i, eval in enumerate(round):
                count += 1
                relevance_sum += eval["Relevance"]
                consistency_sum += eval["Consistency"]
    print(f"Average Relevance: {relevance_sum / count}")
    print(f"Average Consistency: {consistency_sum / count}")
    print(f"Count: {count}")
    ### save results
    mean_results_save_path = os.path.join(data_dir, "gpt-4o_results_r_c.json")
    results_mean = {
        "Relevance": relevance_sum / count,
        "Consistency": consistency_sum / count,
        "Count": count,
    }
    with open(mean_results_save_path, "w") as f:
        json.dump(results_mean, f, indent=4)