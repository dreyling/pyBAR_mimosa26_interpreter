''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''

from numba import njit
import numpy as np


hit_dtype = [('event_number', '<i8'), ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'), ('trigger_status', '<u1'), ('event_status', '<u2')]


def m26_decode_orig(raw, fout, start=0, end=-1):
    print 'DECODE ORIG'
    debug = 1
    n = 10000
    idx = np.zeros(6)
    mframe = [0] * 8
    dlen = [-1] * 6
    idx = [-1] * 6
    numstatus = [0] * 6
    row = [-1] * 6
    dat = np.empty(n, dtype=[('plane', '<u2'), ('mframe', '<u4'), ('x', '<u2'), ('y', '<u2'), ('tlu', '<u2')])
    with open("hit.npy", "wb") as f:
        pass
    raw_i = start
    if end > 0:
        end = min(len(raw), end)
    else:
        end = len(raw)
    hit = 0
    while raw_i < end:
        raw_d = raw[raw_i]
        if hit + 4 >= n:
            print "raw_i", raw_i, "hit", hit, float(raw_i) / end * 100, "% done"
            with open(fout, "ab") as f:
                np.save(f, dat[:hit])
                f.flush()
            hit = 0
        if (0xF0000000 & raw_d == 0x20000000):
            if debug:
                print raw_i, hex(raw_d),
            plane = ((raw_d >> 20) & 0xF)
            mid = plane - 1
            if (0x000FFFFF & raw_d == 0x15555):
                if debug:
                    print "start %d" % mid
                idx[mid] = 0
            elif idx[mid] == -1:
                if debug:
                    print "trash"
            else:
                idx[mid] = idx[mid] + 1
                if debug:
                    print mid, idx[mid],
                if idx[mid] == 1:
                    if debug:
                        print "header"
                    if (0x0000FFFF & raw_d) != (0x5550 | plane):
                        print "header ERROR", hex(raw_d)
                elif idx[mid] == 2:
                    if debug:
                        print "frame lsb"
                    mframe[mid + 1] = (0x0000FFFF & raw_d)
                elif idx[mid] == 3:
                    mframe[plane] = (0x0000FFFF & raw_d) << 16 | mframe[plane]
                    if mid == 0:
                        mframe[0] = mframe[plane]
                    if debug:
                        print "frame", mframe[plane]
                elif idx[mid] == 4:
                    dlen[mid] = (raw_d & 0xFFFF) * 2
                    if debug:
                        print "length", dlen[mid]
                elif idx[mid] == 5:
                    if debug:
                        print "length check"
                    if dlen[mid] != (raw_d & 0xFFFF) * 2:
                        print "dlen ERROR", hex(raw_d)
                elif idx[mid] == 6 + dlen[mid]:
                    if debug:
                        print "tailer"
                    if raw_d & 0xFFFF != 0xaa50:
                        print "tailer ERROR", hex(raw_d)
                elif idx[mid] == 7 + dlen[mid]:
                    dlen[mid] = -1
                    numstatus[mid] = 0
                    if debug:
                        print "frame end"
                    if (raw_d & 0xFFFF) != (0xaa50 | plane):
                        print "tailer ERROR", hex(raw_d)
                else:
                    if numstatus[mid] == 0:
                        if idx[mid] == 6 + dlen[mid] - 1:
                            if debug:
                                print "pass"
                            pass
                        else:
                            numstatus[mid] = (raw_d) & 0xF
                            row[mid] = (raw_d >> 4) & 0x7FF
                            if debug:
                                print "sts", numstatus[mid], "row", row[mid]
                            if raw_d & 0x00008000 != 0:
                                print "overflow", hex(raw_d)
                                break
                    else:
                        numstatus[mid] = numstatus[mid] - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if debug:
                            print "col", col, "num", num
                        for k in range(num + 1):
                            dat[hit] = (plane, mframe[plane], col + k, row[mid], 0)
                            hit = hit + 1
        elif(0x80000000 & raw_d == 0x80000000):
            tlu = raw_d & 0xFFFF
            if debug:
                print hex(raw_d)
            dat[hit] = (7, mframe[0], 0, 0, tlu)
            hit = hit + 1
        raw_i = raw_i + 1
    if debug:
        print "raw_i", raw_i
    if hit == n:
        with open(fout, "ab") as f:
            np.save(f, dat[:hit])
            f.flush()

@njit
def is_mimosa_data(word):  # Check for Mimosa data word
    return 0xF0000000 & word == 0x20000000

@njit
def get_plane_number(word):  # There are 6 planes in the stream, starting from 1; return plane number
    return (word >> 20) & 0xF

@njit
def get_frame_id_high(word):  # Get the frame id from the frame id high word
    return 0x0000FFFF & word

@njit
def get_frame_id_low(word):  # Get the frame id from the frame id low word
    return (0x0000FFFF & word) << 16

@njit
def get_frame_length(word):
    return (word & 0xFFFF) * 2

@njit
def get_row(word):
    return (word >> 4) & 0x7FF

@njit
def get_column(word):
    return (word >> 2) & 0x7FF

@njit
def is_frame_header_high(word):  # Check if frame header high word
    return  0x000FFFFF & word == 0x15555

@njit
def is_frame_header_low(word, plane):  # Check if frame header low word for the actual plane
    return  (0x0000FFFF & word) == (0x5550 | plane)

@njit
def is_frame_tailer_high(word):  # Check if frame header high word
    return  word & 0xFFFF == 0xaa50

@njit
def is_frame_tailer_low(word, plane):  # Check if frame header low word for the actual plane
    return (word & 0xFFFF) == (0xaa50 | plane)

@njit
def get_n_words(word):  # Return the number of data words for the actual row
    return word & 0xF

@njit
def get_n_hits(word):  # Returns the number of hits given by actual column word
    return word & 0x3


def m26_decode(raw_data):
    max_hits_per_event = 1000
    # The order of the data is always START / FRAMW ID / FRAME LENGTH / DATA
    debug = True
    event_number = [0] * 8  # The event counter set by the software counting full events
    frame_id = [0] * 8  # the counter value of the actual frame
    frame_length = [-1] * 6  # the number of data words in the actual frame
    word_index = [-1] * 6  # the word index of per device of the actual frame
    n_words = [0] * 6  # The number of words containing column / row info
    row = [-1] * 6  # the actual readout row (rolling shutter)

    hits = np.zeros(shape=(raw_data.shape[0]), dtype=hit_dtype)
    hit_index = 0

#     hits_buffer = np.zeros((6, max_hits_per_event), dtype=hit_dtype)
#     hit_buffer_index = [0] * 6

    for raw_i in range(raw_data.shape[0]):
        word = raw_data[raw_i]
        if is_mimosa_data(word):
            if debug:
                print hex(word),

            # Check to which plane the data belongs
            actual_plane = get_plane_number(word)
            plane_id = actual_plane - 1  # The actual_plane if the actual word belongs to (0 .. 5)

            # Interpret the word of the actual plane
            if is_frame_header_high(word):  # New event
                if debug:
                    print "start %d" % plane_id
                word_index[plane_id] = 0
            else:
                word_index[plane_id] += 1
                if debug:
                    print plane_id, word_index[plane_id],
                if word_index[plane_id] == 1:  # 1. word should have the header low word
                    if debug:
                        print "header"
                    if not is_frame_header_low(word, actual_plane):
                        print "header ERROR", hex(word)
                elif word_index[plane_id] == 2:  # 2. word should have the frame ID high word
                    if debug:
                        print "frame lsb"
                    frame_id[plane_id + 1] = get_frame_id_high(word)
                elif word_index[plane_id] == 3:  # 3. word should have the frame ID low word
                    frame_id[actual_plane] = get_frame_id_low(word) | frame_id[actual_plane]
                    if plane_id == 0:
                        frame_id[0] = frame_id[plane_id]
                elif word_index[plane_id] == 4:  # 4. word should have the frame length high word
                    frame_length[plane_id] = get_frame_length(word)
                elif word_index[plane_id] == 5:  # 5. word should have the frame length low word (=high word, one data line, the number of words is repeated 2 times)
                    if frame_length[plane_id] != get_frame_length(word):
                        print "frame_length ERROR", hex(word)
                elif word_index[plane_id] == 6 + frame_length[plane_id]:  # Second last word is frame tailer high word
                    if is_frame_tailer_high(word):
                        print "tailer ERROR", hex(word)
                elif word_index[plane_id] == 7 + frame_length[plane_id]:  # First last word is frame tailer low word
                    frame_length[plane_id] = -1
                    n_words[plane_id] = 0
                    if is_frame_tailer_low(word, actual_plane):
                        print "tailer ERROR", hex(word)
                    event_number[plane_id] += 1
                else:  # Column / Row words
                    if n_words[plane_id] == 0:  # First word containing the row info and the number of data words for this row
                        if word_index[plane_id] == 6 + frame_length[plane_id] - 1:  # Always even amount of words or this fill word is used
                            if debug:
                                print "pass"
                        else:
                            n_words[plane_id] = get_n_words(word)
                            row[plane_id] = get_row(word)
                            if debug:
                                print "sts", n_words[plane_id], "row", row[plane_id]
                            if word & 0x00008000 != 0:
                                print "overflow", hex(word)
                                break
                    else:
                        n_words[plane_id] = n_words[plane_id] - 1  # Count down the words
                        n_hits = get_n_hits(word)
                        column = get_column(word)
                        for k in range(n_hits + 1):
                            print (actual_plane, frame_id[plane_id], column + k, row[plane_id], 0)
                            hits[hit_index] = (event_number[plane_id], plane_id, frame_id[plane_id], column + k, row[plane_id], 0, 0)
                            hit_index += 1
#                             dat[hit] = (plane, frame_id[plane], col + k, row[plane_id], 0)
#                             hit = hit + 1
    return hits[:hit_index]



class DataInterpreter(object):

    def __init__(self, max_hits=100000):
        self._max_hits = max_hits  # Maximum hits stored by the data interpreter
        self._hits = np.zeros(self._max_hits, dtype=hit_dtype)

    def interpret_raw_data(self, raw_data):
        m26_decode_orig(raw_data, fout=r'C:\Users\DavidLP\git\pyBAR_mimosa26_interpreter\examples\test.txt')
        print m26_decode(raw_data)
