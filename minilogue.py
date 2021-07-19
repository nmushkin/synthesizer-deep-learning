# Help from https://github.com/jeffkistler/minilogue-editor/blob/master/src/minilogue/midi.js

PARAMETERS_TO_CODE = {
  'NOISE_LEVEL': 33,
  'VCO1_PITCH': 34,
  'VCO2_PITCH': 35,
  'VCO1_SHAPE': 36,
  'VCO2_SHAPE': 37,
  'VCO1_LEVEL': 39,
  'VCO2_LEVEL': 40,
  'CROSS_MOD_DEPTH': 41,
  'VCO2_PITCH_EG_INT': 42,
  'CUTOFF': 43,
  'RESONANCE': 44,
  'CUTOFF_EG_INT': 45,
  'AMP_EG_ATTACK': 16,
  'AMP_EG_DECAY': 17,
  'AMP_EG_SUSTAIN': 18,
  'AMP_EG_RELEASE': 19,
  'EG_ATTACK': 20,
  'EG_DECAY': 21,
  'EG_SUSTAIN': 22,
  'EG_RELEASE': 23,
  'LFO_RATE': 24,
  'LFO_INT': 26,
  'VOICE_MODE_DEPTH': 27,
  'DELAY_HI_PASS_CUTOFF': 29,
  'DELAY_TIME': 30,
  'DELAY_FEEDBACK': 31,
  'VCO1_OCTAVE': 48,
  'VCO2_OCTAVE': 49,
  'VCO1_WAVE': 50,
  'VCO2_WAVE': 51,
  'SYNC': 80,
  'RING': 81,
  'CUTOFF_VELOCITY': 82,
  'CUTOFF_KEYBOARD_TRACK': 83,
  'CUTOFF_TYPE': 84,
  'DELAY_OUTPUT_ROUTING': 88,
  'LFO_TARGET': 56,
  'LFO_EG': 57,
  'LFO_WAVE': 58
}


CODE_TO_PARAMETER = {
  33: 'NOISE_LEVEL',
  34: 'VCO1_PITCH',
  35: 'VCO2_PITCH',
  36: 'VCO1_SHAPE',
  37: 'VCO2_SHAPE',
  39: 'VCO1_LEVEL',
  40: 'VCO2_LEVEL',
  41: 'CROSS_MOD_DEPTH',
  42: 'VCO2_PITCH_EG_INT',
  43: 'CUTOFF',
  44: 'RESONANCE',
  45: 'CUTOFF_EG_INT',
  16: 'AMP_EG_ATTACK',
  17: 'AMP_EG_DECAY',
  18: 'AMP_EG_SUSTAIN',
  19: 'AMP_EG_RELEASE',
  20: 'EG_ATTACK',
  21: 'EG_DECAY',
  22: 'EG_SUSTAIN',
  23: 'EG_RELEASE',
  24: 'LFO_RATE',
  26: 'LFO_INT',
  27: 'VOICE_MODE_DEPTH',
  29: 'DELAY_HI_PASS_CUTOFF',
  30: 'DELAY_TIME',
  31: 'DELAY_FEEDBACK',
  48: 'VCO1_OCTAVE',
  49: 'VCO2_OCTAVE',
  50: 'VCO1_WAVE',
  51: 'VCO2_WAVE',
  80: 'SYNC',
  81: 'RING',
  82: 'CUTOFF_VELOCITY',
  83: 'CUTOFF_KEYBOARD_TRACK',
  84: 'CUTOFF_TYPE',
  88: 'DELAY_OUTPUT_ROUTING',
  56: 'LFO_TARGET',
  57: 'LFO_EG',
  58: 'LFO_WAVE',
}


CONTROL_TYPES = {
  48: 'FOUR_CHOICE',
  49: 'FOUR_CHOICE',
  50: 'THREE_CHOICE',
  51: 'THREE_CHOICE',
  56: 'THREE_CHOICE',
  57: 'THREE_CHOICE',
  58: 'THREE_CHOICE',
  64: 'FOUR_CHOICE',
  65: 'FOUR_CHOICE',
  66: 'THREE_CHOICE',
  67: 'THREE_CHOICE',
  80: 'TWO_CHOICE',
  81: 'TWO_CHOICE',
  82: 'THREE_CHOICE',
  83: 'THREE_CHOICE',
  84: 'TWO_CHOICE',
  88: 'THREE_CHOICE',
  90: 'THREE_CHOICE',
  91: 'THREE_CHOICE',
  92: 'THREE_CHOICE',
}


# Returns valid midi values for different minilogue control types
def control_choices(control_type):
  if (control_type == 'TWO_CHOICE'):
    return [0, 127]
  elif (control_type == 'THREE_CHOICE'):
    return [0, 64, 127]
  elif (control_type == 'FOUR_CHOICE'):
    return [0, 42, 84, 127]
  else:
    return list(range(128))


# Returns a dictionary with controls and default values for the minilogue
def default_control_dict():
  control_dict = {}
  for control_num in PARAMETERS_TO_CODE.values():
    control_type = CONTROL_TYPES.get(control_num)
    control_dict[control_num] = control_choices(control_type)[0]

  # Set sustain to maximum
  control_dict[18] = 127
  # Set EG INT to 0% = halfway
  control_dict[45] = 64
  # Set VCO1 Level to HIGH
  control_dict[39] = 127

  return control_dict