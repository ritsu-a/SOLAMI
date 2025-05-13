import bpy
import numpy as np
import json
import sys
import os
import argparse
sys.path.append(os.getcwd())
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
Arial_path = 'SOLAMI_data/Arial.ttf'

# yellow

# # black
# colors_3 = [
#     (1, 0.1, 0.1, 1),      
#     *[(0., 0., 0.0, 1)] * 21,   
#     *[(0.6, 0.6, 0.6, 1)] * 3,   
#     *[(1., 0.6, 0., 1)] * 15,  
#     *[(0.8, 0., 0., 1)] * 15    
# ]


def get_last_keyframe(scene):
    last_keyframe = 0
    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe_point in fcurve.keyframe_points:
                    frame = keyframe_point.co.x
                    if frame > last_keyframe:
                        last_keyframe = int(frame)
    return last_keyframe


def visualize_skeletons(skeleton_data, start_frames, parents, colors, is_A=True):
    if skeleton_data[0].shape[1] == 52:
        for i, skeleton in enumerate(skeleton_data):
            eyes = np.repeat(skeleton[:, 15:16], 3, axis=1)
            skeleton_data[i] = np.concatenate([skeleton[:, :22], eyes, skeleton[:, 22:]], axis=1)
    joints = []
    for i in range(55):
        if i < 25:
            joint_scale = 0.015
        else:
            joint_scale = 0.005
        bpy.ops.mesh.primitive_uv_sphere_add(radius=joint_scale, location=skeleton_data[0][0, i, :])
        joint = bpy.context.view_layer.objects.active
        mat = bpy.data.materials.new(name=f'Joint_{i}_Mat')
        mat.diffuse_color = colors[i]
        joint.data.materials.append(mat)
        joints.append(joint)


    bones = []
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            continue
        start_joint = joints[parent_idx]
        end_joint = joints[i]
        start_loc = start_joint.location
        end_loc = end_joint.location
        mid_loc = (start_loc + end_loc) / 2
        bone_length = (end_loc - start_loc).length

        bpy.ops.mesh.primitive_cube_add(size=0.8, location=mid_loc)
        bone = bpy.context.view_layer.objects.active
        if i < 25:
            bone_scale = 0.02
        else:
            bone_scale = 0.01
        bone.scale = (bone_scale, bone_scale, bone_length / 1.2)
        bone.rotation_mode = 'QUATERNION'
        direction = end_loc - start_loc
        bone.rotation_quaternion = direction.to_track_quat('Z', 'Y')
        bone.location = mid_loc
        bones.append(bone)

        bone.data.materials.append(start_joint.data.materials[0])

    
    for idx in range(len(skeleton_data)):
        if is_A:
            id = idx * 2
        else:
            id = idx * 2 + 1
        start_frame = start_frames[id]
        end_frame = start_frames[id] + skeleton_data[idx].shape[0]

        for frame in range(start_frame, end_frame, 1):
            bpy.context.scene.frame_set(int(frame))
            for i, joint in enumerate(joints):
                joint.location = skeleton_data[idx][frame - start_frame, i, :]
                joint.keyframe_insert(data_path="location", index=-1, frame=frame)
            for i, bone in enumerate(bones):
                parent_idx = parents[i + 1]
                if parent_idx == -1:
                    continue
                start_joint = joints[parent_idx]
                end_joint = joints[i + 1]
                start_loc = start_joint.location
                end_loc = end_joint.location
                mid_loc = (start_loc + end_loc) / 2
                bone.location = mid_loc
                bone_length = (end_loc - start_loc).length
                if i < 25:
                    bone_scale = 0.02
                else:
                    bone_scale = 0.01
                bone.scale = (bone_scale, bone_scale, bone_length / 1.2)
                direction = end_loc - start_loc
                bone.rotation_quaternion = direction.to_track_quat('Z', 'Y')
                bone.keyframe_insert(data_path="location", index=-1, frame=frame)
                bone.keyframe_insert(data_path="scale", index=-1, frame=frame)
                bone.keyframe_insert(data_path="rotation_quaternion", index=-1, frame=frame)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = start_frames[-1]

    print("Skeleton animation set over!")


class ArgumentParserForBlender(argparse.ArgumentParser):
    """This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because it
    will try to process the script's -a and -b flags:

    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns an
        empty list)."""
        try:
            idx = sys.argv.index('--')
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            print(f'Get argv error: {e}')
            return []

    # overrides superclass
    def parse_args(self):
        """This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before.

        See the docstring of the class for usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def setup_parser():
    parser = ArgumentParserForBlender(
        description='Visualize a list of SMPL-X in one scene')
    # parser.add_argument(
    #     '--filelist', nargs='+', help='SMPL-X filelist', required=True)
    parser.add_argument(
        '--json_path',
        type=str,
        default='/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1/sim1__1200_0_if_sc.json')
    parser.add_argument(
        '--engine_type',
        type=str,
        help='Engine types for blender renderer',
        default='EEVEE')
    args = parser.parse_args()
    return args


def process_A_motion(A_motion_list):
    R = np.array([[-1, 0, 0],
                  [0, 1, 0],
                    [0, 0, -1]])
    t = np.array([0, 0, 1.5])
    new_motion = []
    for motion in A_motion_list:
        new_motion.append(np.dot(motion, R.T) + t)
    return new_motion



if __name__ == '__main__':
    args = setup_parser()
    bpy.ops.object.select_all(action='DESELECT')
    # bpy.ops.wm.read_factory_settings(use_empty=True)
    if 'Cube' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)
    if 'Empty' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Empty'], do_unlink=True)
    if 'Light' not in bpy.data.objects.keys():
        bpy.ops.object.light_add(type='POINT')
        light = bpy.context.object
        light.name = 'Light'
    if 'Camera' not in bpy.data.objects.keys():
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW')
    
    json_path = args.json_path
    print(json_path)
    with open(json_path, 'r') as f:
        conv_data = json.load(f)
    
    m_dataset_items = {}
    item_paths = [
        "SOLAMI_data/HumanML3D/dataset_items.json",
        "SOLAMI_data/DLP-MoCap/dataset_items.json",
        "SOLAMI_data/Inter-X/dataset_items.json",
    ]
    for item_path in item_paths:
        with open(item_path, 'r') as f:
            m_dataset_items.update(json.load(f))
            
    motion_dirs = {
        'dlp': 'SOLAMI_data/DLP-MoCap',
        'humanml3d': 'SOLAMI_data/HumanML3D',
        'interx': 'SOLAMI_data/Inter-X',
    }
    align_dir = "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1_align"
    audio_dir = "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1_audio"
    # align_path = os.path.join(align_dir, os.path.basename(json_path))
    # with open(align_path, 'r') as f:
    #     aligns = json.load(f)

    aligns = {}
    basename = os.path.basename(json_path).split('.')[0]
    audio_dir = os.path.join(audio_dir, basename)
    p_time = 0
    
    A_motion_list, B_motion_list = [], []
    # A_motion_frames, B_motion_frames = [0, ], [0, ]
    all_motion_frames = [0, ]
    A_text_list, B_text_list = [], []
    
    for round, dialog in conv_data[0]['dialogs'].items():
        audio_tmp = AudioFileClip(os.path.join(audio_dir, f'{round}.wav'))
        audio_length = audio_tmp.duration
        motion_dataset_name = dialog['action_dataset']
        motion_path_tmp = m_dataset_items[dialog['action_index']]['motion_joints_path']
        motion_path = os.path.join(motion_dirs[motion_dataset_name], motion_path_tmp)
        motion = np.load(motion_path)
        start_frame = m_dataset_items[dialog['action_index']]['start_frame']
        end_frame = m_dataset_items[dialog['action_index']]['end_frame']
        motion = motion[start_frame:end_frame]
        
        motion_length = motion.shape[0] / 30
        max_duration = max(audio_length, motion_length)
        p_time += max_duration + 1
        aligns[round] = p_time
        
        if dialog['role'] == 'A':
            A_motion_list.append(motion)
            # A_motion_frames.append(int(aligns[round] * 30))
            A_text_list.append(dialog['speech'])
        else:
            B_motion_list.append(motion)
            # B_motion_frames.append(int(aligns[round] * 30))
            B_text_list.append(dialog['speech'])
        all_motion_frames.append(int(aligns[round] * 30))
    
    
    A_motion_list = process_A_motion(A_motion_list)
    
    lens = 6
    
    A_motion_list, B_motion_list = A_motion_list[:lens], B_motion_list[:lens]
    # A_motion_frames, B_motion_frames = A_motion_frames[:lens+1], B_motion_frames[:lens+1]
    A_text_list, B_text_list = A_text_list[:lens], B_text_list[:lens]
    
    
    colors_1 = [
            (1, 0.1, 0.1, 1),       
            *[(1., 1., 0.0, 1)] * 21,  
            *[(0.6, 0.6, 0.6, 1)] * 3,    
            *[(1., 0.6, 1., 1)] * 15,  
            *[(0.6, 1., 1., 1)] * 15    
        ]

    # pink
    colors_2 = [
                (1, 0.1, 0.1, 1),       
                *[(1., 0., 1.0, 1)] * 21,   
                *[(0.6, 0.6, 0.6, 1)] * 3,   
                *[(1., 0.6, 1., 1)] * 15,  
                *[(0.6, 1., 1., 1)] * 15   
            ]

    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
            35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
            53]
    
    visualize_skeletons(A_motion_list, all_motion_frames, parents, colors_1, is_A=True)
    visualize_skeletons(B_motion_list, all_motion_frames, parents, colors_2, is_A=False)
    
    
    bpy.data.scenes['Scene'].view_settings.view_transform = 'Standard'

    # bpy.data.worlds['World'].node_tree.\
    #     nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
    last_keyframe = get_last_keyframe(bpy.context.scene)

    # bpy.context.scene.render.engine = 'CYCLES'

    # Set the output file format to FFmpeg video
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

    # Set the output format to MPEG-4
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0

    # bpy.data.objects["SMPLX-female"].hide_render= True
    # bpy.data.objects["SMPLX-mesh-female"].hide_render = True
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    # bpy.context.scene.render.engine = 'CYCLES'

    # bpy.data.objects["Camera"].location[0] = 3.63
    # bpy.data.objects["Camera"].location[1] = 2.9
    # bpy.data.objects["Camera"].location[2] = 1.06
    
    if "Camera" in bpy.data.objects:
        camera_obj = bpy.data.objects["Camera"]
        bpy.context.scene.camera = camera_obj
        camera_obj.location = (3.63, 2.9, 1.06)
        camera_obj.rotation_euler = (-57.23 / 90 * 1.5708, 77.13 / 90 * 1.5708, 331.85 / 90 * 1.5708)
        camera_obj.data.lens = 20
    else:
        raise RuntimeError("No camera named 'Camera' found in the scene.")
    
    
    bpy.data.scenes["Scene"].view_settings.view_transform = 'Standard'
    # bpy.data.worlds["World"].color = (1, 1, 1)

    video_dir = "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1_rendered"
    os.makedirs(video_dir, exist_ok=True)
    video_path_tmp = os.path.join(align_dir, os.path.basename(json_path).split('.')[0] + '_rendered.mp4')
    video_path_final = os.path.join(video_dir, os.path.basename(json_path).split('.')[0] + '.mp4')
    # render to video output path
    bpy.context.scene.render.filepath = video_path_tmp

    # Set the start and end frames for the animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = int(last_keyframe)

    # Render the animation
    bpy.ops.render.render(animation=True)

    video = VideoFileClip(video_path_tmp)

    final_audios = []

    for round, dialog in conv_data[0]['dialogs'].items():
        audio_path = os.path.join(audio_dir, f'{round}.wav')
        audio = AudioFileClip(audio_path)
        audio = audio.set_start(all_motion_frames[int(round)] / 30)
        final_audios.append(audio)
    final_audio = CompositeAudioClip(final_audios)
    final_audio = final_audio.subclip(0, video.duration)
    video = video.set_audio(final_audio)

    text_colors = {}
    text_colors['A'] = "rgb({},{},{})".format(int(colors_1[2][0]*255), int(colors_1[2][1]*255), int(colors_1[2][2]*255))
    text_colors['B'] = "rgb({},{},{})".format(int(colors_2[2][0]*255), int(colors_2[2][1]*255), int(colors_2[2][2]*255))

    text_clips = []
    video_duration = video.duration
    text_clips = []
    for i, text in enumerate(A_text_list):
        text_duration = (all_motion_frames[2*i+1] - all_motion_frames[2*i]) / 30 - 1
        txt_clip = TextClip("A: " + text, fontsize=35, color=text_colors['A'], font=Arial_path).set_position((50, 50)).set_duration(text_duration).set_start(all_motion_frames[2*i] / 30)
        text_clips.append(txt_clip)
    for i, text in enumerate(B_text_list):
        text_duration = (all_motion_frames[2*i+2] - all_motion_frames[2*i+1]) / 30 -1
        txt_clip = TextClip("B: " + text, fontsize=35, color=text_colors['B'], font=Arial_path).set_position((50, 50 + 80)).set_duration(text_duration).set_start(all_motion_frames[2*i+1] / 30)
        text_clips.append(txt_clip)
   
    final_clip = CompositeVideoClip([video] + text_clips)

   
    final_clip.write_videofile(video_path_final, codec='libx264', audio_codec="aac", fps=30)

    
    video.close()
    for txt_clip in text_clips:
        txt_clip.close()
    