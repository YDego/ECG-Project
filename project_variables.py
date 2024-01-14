SIGNAL_LEN_FOR_LUDB = 10
SIGNAL_LEN_FOR_QT = 900
SIGNAL_LEN_FOR_MIT = 1806
WINDOW_SIZE_FOR_T_PEAKS = 0.070
WINDOW_SIZE_FOR_P_PEAKS = 0.030

ALL_SIGNALS_INPUT = ['l', 'q', 'm']

DATASETS = {
    'l': 'ludb',
    'q': 'qt',
    'm': 'mit'
}

NUM_OF_SETS = {
    'l': 201,
    'q': 106,
    'm': 46
}

SIGNAL_LEN = {
    'l': SIGNAL_LEN_FOR_LUDB,
    'q': SIGNAL_LEN_FOR_QT,
    'm': SIGNAL_LEN_FOR_MIT
}

# plot variables
MARKER_DICT = {
    '(': '<',
    ')': '>',
    'p': 'X',
    'N': 'X',
    'n': 'o',
    't': 'X'
}

COLOR_DICT = {
    '(': 'k',
    ')': 'k',
    'p': 'b',
    'N': 'g',
    'n': 'g',
    't': 'r'
}

MARKER_TITLE_DICT = {
    '(': 'Opening point',
    ')': 'Closing point',
    'p': 'P peak',
    'N': 'R peak',
    'n': 'R peak',
    't': 't peak'
}