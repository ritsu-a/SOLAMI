# image_prefix = "👀"
speech_prefix = "🗣️"
# music_prefix = "🎶"
audio_prefix = "👂"
# body_prefix =  "🕺"
motion_prefix =  "🕺"
hand_prefix = "🤚"
trans_prefix = "🔀"
#body_prefix =  ""  #"🕺"


start_of_image, end_of_image = '<soim>', '<eoim>'
start_of_speech, end_of_speech = '<sosp>', '<eosp>'
start_of_music, end_of_music = '<somu>', '<eomu>'
start_of_audio, end_of_audio = '<soau>', '<eoau>'
start_of_body, end_of_body = '<sobo>', '<eobo>'
start_of_motion, end_of_motion = '<somo>', '<eomo>'
start_of_hand, end_of_hand = '<soha>', '<eoha>'


image_vocab_size=8192
speech_vocab_size=1024
music_codebook_size=2048
music_codebook_num=4
music_vocab_size=music_codebook_size * music_codebook_num
audio_codebook_size=1024
audio_codebook_num=4
audio_vocab_size=audio_codebook_size * audio_codebook_num

body_codebook_num=512
motion_codebook_size=512
motion_codebook_num=1
motion_vocab_size = motion_codebook_size * motion_codebook_num


hand_codebook_num=512
trans_codebook_num=256

modal_special_str = {
    # "image":{
    #     "prefix": image_prefix,
    #     "sos": start_of_image,
    #     "eos": end_of_image,
    #     "vocab_size": image_vocab_size
    # },
    "speech":{
        "prefix": speech_prefix,
        "sos": start_of_speech,
        "eos": end_of_speech,
        "vocab_size": speech_vocab_size
    },
    # "music":{
    #     "prefix": music_prefix,
    #     "sos": start_of_music,
    #     "eos": end_of_music,
    #     "vocab_size": music_vocab_size
    # },
    "motion":{
        "prefix": motion_prefix,
        "sos": start_of_motion,
        "eos": end_of_motion,
        "vocab_size": motion_vocab_size,
    },
    # "body":{
    #     "prefix": body_prefix,
    #     "sos": start_of_body,
    #     "eos": end_of_body,
    #     "vocab_size": body_codebook_num,
    # },
    # "hand":{
    #     "prefix": hand_prefix,
    #     "sos": start_of_hand,
    #     "eos": end_of_hand,
    #     "vocab_size": hand_codebook_num,
    # },
    # "trans": {
    #     'prefix': trans_prefix,
    #     "sos": '',
    #     "eos": '',
    #     "vocab_size": trans_codebook_num,
    # }
}

# modal_special_str = {
#     "body":{
#         "prefix": body_prefix,
#         "sos": start_of_body,
#         "eos": end_of_body,
#         "vocab_size": body_codebook_num,
#     },
#     "speech":{
#         "prefix": speech_prefix,
#         "sos": start_of_speech,
#         "eos": end_of_speech,
#         "vocab_size": speech_vocab_size
#     },
#     "hand":{
#         "prefix": hand_prefix,
#         "sos": start_of_hand,
#         "eos": end_of_hand,
#         "vocab_size": hand_codebook_num,
#     },
#     "trans": {
#         'prefix': trans_prefix,
#         "sos": '',
#         "eos": '',
#         "vocab_size": trans_codebook_num,
#     }
# }

    # "audio":{
    #     "prefix": audio_prefix,
    #     "sos": start_of_audio,
    #     "eos": end_of_audio,
    #     "vocab_size": audio_vocab_size
    # }


def modality_tokens_to_string(tokens, modality="image"):
    """
    Convert visual tokens to a single string with prefix and postfix.
    """
    prefix=modal_special_str[modality]["prefix"]
    start=modal_special_str[modality]["sos"]
    end=modal_special_str[modality]["eos"]
    
    # if modality == "music":
    #     # music tokens are 2-dim array
    #     # Convert each token to its corresponding string representation
    #     tokens_str = []
    #     # 第0层维持原值，第1层每个值+music_codebook_size，第2层每个值+music_codebook_size*2，第3层每个值+music_codebook_size*3    
    #     # 按层堆叠，依次加入第一帧的四层，第二帧的四层，依次类推
    #     for idx in range(len(tokens[0])):
    #         for layer_idx in range(len(tokens)):
    #             tokens_str.append(f"<{prefix}{tokens[layer_idx][idx] + music_codebook_size * layer_idx}>")          
    # elif modality == "audio":
    #     # audio tokens are 2-dim array
    #     # Convert each token to its corresponding string representation
    #     tokens_str = []
    #     # 第0层维持原值，第1层每个值+music_codebook_size，第2层每个值+music_codebook_size*2，第3层每个值+music_codebook_size*3    
    #     # 按层堆叠，依次加入第一帧的四层，第二帧的四层，依次类推
    #     for idx in range(len(tokens[0])):
    #         for layer_idx in range(len(tokens)):
    #             tokens_str.append(f"<{prefix}{tokens[layer_idx][idx] + audio_codebook_size * layer_idx}>")


    # if modality == "motion":
    #     # Convert each token to its corresponding string representation
    #     tokens_str = []
    #     # 第0层维持原值，第1层每个值+motion_codebook_size
    #     # 按层堆叠，依次加入第一帧的四层，第二帧的四层，依次类推
    #     for idx in range(len(tokens[0])):
    #         for layer_idx in range(len(tokens)):
    #             tokens_str.append(f"<{prefix}{tokens[layer_idx][idx] + motion_codebook_size * layer_idx}>")       
    
    if False:
        pass

    else:
        # Convert each token to its corresponding string representation
        tokens_str = [f"<{prefix}{token}>" for token in tokens]
        # Join the token strings and add <soim> at the beginning and <eoim> at the end
    return start + "".join(tokens_str) + end


if __name__ == "__main__":
    print(modality_tokens_to_string([1, 32, 23], modality="image"))
    print(modality_tokens_to_string([1, 32, 23], modality="speech"))
    print(modality_tokens_to_string([[1, 32, 23], [0, 32, 23], [0, 32, 23], [0, 32, 23]], modality="music"))
    print(modality_tokens_to_string([[1, 32, 23], [0, 32, 23], [0, 32, 23], [0, 32, 23]], modality="audio"))
