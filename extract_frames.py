import ffmpeg
import subprocess
import argparse
import sys, os
from pathlib import Path


BASE_DIR = Path('..')

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utility_functions import create_output_dir_if_req, clean_output_dir

parser = argparse.ArgumentParser(description='Frame Extractor')

parser.add_argument('--vid_path', default='example.avi',
                    type=str, help='Path to the video from which the frames will be extracted')
parser.add_argument('--output_path', default='sampled_frames/orig',
                    type=str, help='Path to where the frames should be extracted')
parser.add_argument('--nb_frames', default=30,
                    type=int, help='Number of frames to be extracted from the video')

args = parser.parse_args()


def sample_frames(vid_path, output_path, nb_frames):
    try:
        probe_result = ffmpeg.probe(vid_path)
        tot_nb_frames_vid = int(probe_result.get('streams')[0].get('nb_frames'))
        print(tot_nb_frames_vid)
        ratio_total_sampled = tot_nb_frames_vid / nb_frames
        vid_frame_rate_r = probe_result.get('streams')[0].get('r_frame_rate')
        num1, num2 = vid_frame_rate_r.split('/')[0], vid_frame_rate_r.split('/')[-1]
        vid_frame_rate = float(num1) / float(num2)
        new_vid_frame_rate = vid_frame_rate / ratio_total_sampled
        new_vid_frame_rate_r = str(new_vid_frame_rate) + '/' + '1'
        print(new_vid_frame_rate_r)
        # sample all frames if frames_tb_sampled > total_vid_frames
        if ratio_total_sampled < 1:
            (
                ffmpeg
                .input(vid_path)
                .filter('select', n=nb_frames)
                #.filter('nth', 'n={}'.format(nb_frames))
                .output(f'{output_path}/frames_%03d.png', start_number=0)
                .run(capture_stderr=True)
            )
        # sample at regular stride within intervals
        else:
            command_tb_run = f'ffmpeg -i {vid_path} -vf fps={new_vid_frame_rate_r} {output_path}/frames_%03d.png'
            print(command_tb_run)
            try:
                subp_output = subprocess.run(command_tb_run  # , shell=True
                                            )
            except Exception as exc:
                print(
                    f'[An exception of type {type(exc).__name__} occurred. Arguments:\n{exc.args!r}]')
                print(f'STDERR of subprocess:\n{subp_output.stderr}')
            
    except ffmpeg.Error as e:
        out_msg = e.stdout
        if out_msg is not None:
            out_msg = e.stdout.decode('utf8')
            print('stdout:', out_msg)
        err_msg = e.stderr
        if err_msg is not None:
            err_msg = e.stderr.decode('utf8')
            print('stderr:', err_msg)
    pass

if __name__ == '__main__':
    vid_path = BASE_DIR / args.vid_path
    output_path = BASE_DIR / args.output_path
    create_output_dir_if_req(output_path)
    clean_output_dir(output_path)
    sample_frames(vid_path, output_path, args.nb_frames)