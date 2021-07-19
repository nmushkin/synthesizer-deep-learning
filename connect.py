import json
from posix import listdir
from time import sleep, time
from random import randint, choice
from json import dump, load
from os import path

import mido
import numpy as np
import sounddevice as sd
from torchaudio.transforms import Spectrogram
from PIL import Image

import minilogue

RECORD_DURATION = 0.5  # seconds
SAMPLERATE = 48000

CURRENT_PROGRAM_DATA_DUMP_REQUEST = 0x10

ACTIVE_CONTROLS = [
    # 'NOISE_LEVEL',
    'VCO1_PITCH',
    # 'VCO2_PITCH',
    'VCO1_SHAPE',
    # 'VCO2_SHAPE',
    # 'VCO1_LEVEL',
    # 'VCO2_LEVEL',
    # 'CROSS_MOD_DEPTH',
    # 'VCO2_PITCH_EG_INT',
    'CUTOFF',
    'RESONANCE',
    # 'CUTOFF_EG_INT',
    # 'AMP_EG_ATTACK',
    # 'AMP_EG_DECAY',
    # 'AMP_EG_SUSTAIN',
    # 'AMP_EG_RELEASE',
    # 'EG_ATTACK',
    # 'EG_DECAY',
    # 'EG_SUSTAIN',
    # 'EG_RELEASE',
    # 'LFO_RATE',
    # 'LFO_INT',
    # 'VOICE_MODE_DEPTH',
    # 'DELAY_HI_PASS_CUTOFF',
    # 'DELAY_TIME',
    # 'DELAY_FEEDBACK',
    'VCO1_OCTAVE',
    # 'VCO2_OCTAVE',
    'VCO1_WAVE',
    # 'VCO2_WAVE',
    # 'SYNC',
    # 'RING',
    # 'CUTOFF_VELOCITY',
    # 'CUTOFF_KEYBOARD_TRACK',
    # 'CUTOFF_TYPE',
    # 'DELAY_OUTPUT_ROUTING',
    # 'LFO_TARGET',
    # 'LFO_EG',
    # 'LFO_WAVE'
]


ACTIVE_CONTROL_CODES = {
    minilogue.PARAMETERS_TO_CODE[c]: 0 for c in ACTIVE_CONTROLS
}


DATA_DUMP_MESSAGE = [
    0xf0, 0x42, 0x30, 0x00, 0x01, 0x2c, CURRENT_PROGRAM_DATA_DUMP_REQUEST, 0xf7
    ]


# Stores a sample (waveform) with the controls that produced it
def write_control_file(control_dict, sample_array, sample_id, directory):
    sample_json = {
        'id': sample_id,
        'controls': control_dict,
        'sample': sample_array.tolist()
    }

    with open(path.join(directory, f'{sample_id}.json'), 'w') as write_fp:
        dump(sample_json, write_fp)


def write_sample_image(sample_array, sample_id, directory):
    sound_image = Image.fromarray(sample_array).convert('RGB')
    sound_image.save(path.join(directory, f'{sample_id}.jpeg'))


# Plays a note on the synthesizer and records the sample
def play_and_record(outport):
    outport.send(mido.Message('note_on', note=60, channel=0))
    myrecording = sd.rec(int(RECORD_DURATION * SAMPLERATE), channels=1, blocking=True, device=2)
    sd.play(myrecording, samplerate=SAMPLERATE, blocking=False)
    sample_image = myrecording.reshape(int(RECORD_DURATION * SAMPLERATE))
    outport.send(mido.Message('note_off', note=60, channel=0))
    
    return sample_image


# Sends a control state to the synth, plays that sound, and records the sample
def make_sample(outport, sample_id, sound_directory, control_directory):
    start = time()
    # for control in ACTIVE_CONTROL_CODES:
    control = choice(list(ACTIVE_CONTROL_CODES.keys()))
    new_value = random_control_value(control)
    ACTIVE_CONTROL_CODES[control] = new_value
    send_control_states(outport, {control: new_value})
    sample = play_and_record(outport)
    write_sample_image(sample, sample_id, sound_directory)
    write_control_file(ACTIVE_CONTROL_CODES, sample, sample_id, control_directory)
    print(time() - start)


# Sends a dict of control values to the synth over midi
def send_control_states(outport, control_values: dict):
    for control, value in control_values.items():
        print(control, value)
        outport.send(mido.Message('control_change', control=control, value=value))
        sleep(.01)


# Generates a number of synth sound samples with control data
def generate_samples(num_samples, sound_directory, control_directory):
    inport, outport = init_interfaces()
    control_defaults = minilogue.default_control_dict()
    print(control_defaults)
    send_control_states(outport, control_defaults)
    for sample_id in range(1, num_samples + 1):
        make_sample(outport, sample_id + 10000, sound_directory, control_directory)


# Starts up the audio and midi interfaces, returning the midi ports
def init_interfaces():

    def msg_callback(msg):
        if msg.type != 'clock':
            print(msg, msg.hex())
    
    print(mido.get_output_names()) # To list the output ports
    print(mido.get_input_names()) # To list the input ports
    print(sd.query_devices()) # Lists sound interfaces

    sd.default.samplerate = SAMPLERATE

    inport = mido.open_input('minilogue KBD/KNOB', callback=msg_callback)
    outport = mido.open_output('minilogue SOUND')

    return inport, outport


# Gets a valid random control value based on the control number
def random_control_value(control_num):
    control_type = minilogue.CONTROL_TYPES.get(control_num)
    control_coices = minilogue.control_choices(control_type)
    return control_coices[randint(0, len(control_coices) - 1)]


generate_samples(20000, './data/sound_images', './data/control_data')

# def spectrograms_from_waves(from_folder, to_folder):
#     for json_fname in listdir(from_folder):
#         if '.json' not in json_fname:
#             continue
#         json_fname = path.join(from_folder, json_fname)
#         json_fp = open(json_fname, 'r')
#         json_data = load(json_fp)
#         json_fp.close()
#         sample_id = json_data['id']
#         param_values = json_data['controls']
#         wave_values = json_data['sample']
#         Spectrogram()

