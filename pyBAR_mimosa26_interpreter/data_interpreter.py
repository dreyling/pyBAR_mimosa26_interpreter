''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''
import os
from numba import njit
import numpy as np
import tables as tb
import logging
import progressbar
from matplotlib.backends.backend_pdf import PdfPages


hit_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'), ('trigger_status', '<u1'), ('event_status', '<u2')])


# Event error codes
NO_ERROR = 0  # No error
MULTIPLE_TRG_WORD = 1  # Event has more than one trigger word
NO_TRG_WORD = 2  # Some hits of the event have no trigger word
DATA_ERROR = 4  # Event has data word combinations that do not make sense (tailor at wrong position, not increasing frame counter ...)
EVENT_INCOMPLETE = 8  # Data words are missing (e.g. tailor header)
UNKNOWN_WORD = 16  # Event has unknown words
UNEVEN_EVENT = 32  # Event has uneven amount of data words
TRG_ERROR = 64  # A trigger error occured
TRUNC_EVENT = 128  # Event had to many hits and was truncated
TDC_WORD = 256  # Event has a TDC word
MANY_TDC_WORDS = 512  # Event has more than one valid TDC word
TDC_OVERFLOW = 1024  # Event has TDC word indicating a TDC overflow
NO_HIT = 2048  # vents without any hit, usefull for trigger number debugging


def m26_decode_orig(raw, fout, start=0, end=-1):
    print('DECODE ORIG')
    debug = 0
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
            print("raw_i", raw_i, "hit", hit, float(raw_i) / end * 100, "% done")
            with open(fout, "ab") as f:
                np.save(f, dat[:hit])
                f.flush()
            hit = 0
        if (0xF0000000 & raw_d == 0x20000000):
            if debug:
                print(raw_i, hex(raw_d),)
            plane = ((raw_d >> 20) & 0xF)
            mid = plane - 1
            if (0x000FFFFF & raw_d == 0x15555):
                if debug:
                    print("start %d" % mid)
                idx[mid] = 0
            elif idx[mid] == -1:
                if debug:
                    print("trash")
            else:
                idx[mid] = idx[mid] + 1
                if debug:
                    print(mid, idx[mid],)
                if idx[mid] == 1:
                    if debug:
                        print("header")
                    if (0x0000FFFF & raw_d) != (0x5550 | plane):
                        print("header ERROR", hex(raw_d))
                elif idx[mid] == 2:
                    if debug:
                        print("frame lsb")
                    mframe[mid + 1] = (0x0000FFFF & raw_d)
                elif idx[mid] == 3:
                    mframe[plane] = (0x0000FFFF & raw_d) << 16 | mframe[plane]
                    if mid == 0:
                        mframe[0] = mframe[plane]
                    if debug:
                        print("frame", mframe[plane])
                elif idx[mid] == 4:
                    dlen[mid] = (raw_d & 0xFFFF) * 2
                    if debug:
                        print("length", dlen[mid])
                elif idx[mid] == 5:
                    if debug:
                        print("length check")
                    if dlen[mid] != (raw_d & 0xFFFF) * 2:
                        print("dlen ERROR", hex(raw_d))
                elif idx[mid] == 6 + dlen[mid]:
                    if debug:
                        print("tailer")
                    if raw_d & 0xFFFF != 0xaa50:
                        print("tailer ERROR", hex(raw_d))
                elif idx[mid] == 7 + dlen[mid]:
                    dlen[mid] = -1
                    numstatus[mid] = 0
                    if debug:
                        print("frame end")
                    if (raw_d & 0xFFFF) != (0xaa50 | plane):
                        print("tailer ERROR", hex(raw_d))
                else:
                    if numstatus[mid] == 0:
                        if idx[mid] == 6 + dlen[mid] - 1:
                            if debug:
                                print("pass")
                            pass
                        else:
                            numstatus[mid] = (raw_d) & 0xF
                            row[mid] = (raw_d >> 4) & 0x7FF
                            if debug:
                                print("sts", numstatus[mid], "row", row[mid])
                            if raw_d & 0x00008000 != 0:
                                print("overflow", hex(raw_d))
                                break
                    else:
                        numstatus[mid] = numstatus[mid] - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if debug:
                            print("col", col, "num", num)
                        for k in range(num + 1):
                            dat[hit] = (plane, mframe[plane], col + k, row[mid], 0)
                            hit = hit + 1
        elif(0x80000000 & raw_d == 0x80000000):
            tlu = raw_d & 0xFFFF
            if debug:
                print(hex(raw_d))
            dat[hit] = (7, mframe[0], 0, 0, tlu)
            hit = hit + 1
        raw_i = raw_i + 1
    if debug:
        print("raw_i", raw_i)
    if hit == n:
        with open(fout, "ab") as f:
            np.save(f, dat[:hit])
            f.flush()

    return dat


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
    return 0x000FFFFF & word == 0x15555


@njit
def is_frame_header_low(word, plane):  # Check if frame header low word for the actual plane
    return (0x0000FFFF & word) == (0x5550 | plane)


@njit
def is_frame_tailer_high(word):  # Check if frame header high word
    return word & 0xFFFF == 0xaa50


@njit
def is_frame_tailer_low(word, plane):  # Check if frame header low word for the actual plane
    return (word & 0xFFFF) == (0xaa50 | plane)


@njit
def get_n_words(word):  # Return the number of data words for the actual row
    return word & 0xF


@njit
def get_n_hits(word):  # Returns the number of hits given by actual column word
    return word & 0x3


@njit
def has_overflow(word):
    return word & 0x00008000 != 0


@njit
def is_trigger_word(word):
    return 0x80000000 & word == 0x80000000


@njit
def get_trigger_number(word):  # Returns the number of hits given by actual column word
    return word & 0xFFFF


@njit
def add_event_status(plane_id, event_status, status_code):
    # print 'ADD EVENT STATUS', status_code
    event_status[plane_id] |= status_code


@njit
def finish_event(plane_id, hits_buffer, hit_buffer_index, event_status, hits, hit_index):  # Append buffered hits to hit object
    for i_hit in range(hit_buffer_index):  # Loop over buffered hits
        hits[hit_index] = hits_buffer[plane_id, i_hit]
        hits[hit_index].event_status = event_status
        hit_index += 1
    return hit_index  # Return actual hit index; needed to append correctly at next call of finish_event


class RawDataInterpreter(object):

    def __init__(self, max_hits_per_event=1000, debug=False):
        self.max_hits_per_event = max_hits_per_event
        self.debug = debug
        self.reset()

    def reset(self):  # Reset variables
        # Per frame variables
        self.frame_id = [0] * 8  # The counter value of the actual frame
        self.last_frame_id = [-1] * 8  # The counter value of the last frame
        self.frame_length = [-1] * 6  # The number of data words in the actual frame
        self.word_index = [-1] * 6  # The word index per device of the actual frame
        self.n_words = [0] * 6  # The number of words containing column / row info
        self.row = [-1] * 6  # the actual readout row (rolling shutter)

        # Per event variables
        self.hits_buffer = np.zeros((6, self.max_hits_per_event), dtype=hit_dtype)  # Buffers actual event hits, needed since raw data is analyzed in chunks
        self.hit_buffer_index = [0] * 6  # Hit buffer index for each plane; needed to append hits
        self.event_status = np.zeros(shape=(6, ), dtype=np.uint16)  # Actual event status for each plane
        self.event_number = [-1] * 8  # The event counter set by the software counting full events for each plane
        self.trigger_number = -1  # The actual event trigger number

    def build_hits(self, raw_data):
        # The order of the data should be always START / FRAMEs ID / FRAME LENGTH / DATA
        # Since the clock is the same for each plane; the order is START plane 1, START plane 2, ...

        hits = np.zeros(shape=(raw_data.shape[0]), dtype=hit_dtype)  # Result hits array
        hit_index = 0  # Pointer to actual hit in resul hit arrray; needed to append hits every event

        for raw_i in range(raw_data.shape[0]):
            word = raw_data[raw_i]  # Actual raw data word
            if is_mimosa_data(word):  # There can be not mimosa related data words (from FE-I4)
                if self.debug:
                    print(raw_i, hex(word),)

                # Check to which plane the data belongs
                actual_plane = get_plane_number(word)
                plane_id = actual_plane - 1  # The actual_plane if the actual word belongs to (0 .. 5)

                # Interpret the word of the actual plane
                if is_frame_header_high(word):  # New event for actual plane; events are aligned at this header
                    # Finish old event
                    if self.event_number[plane_id] >= 0:  # First event 0 should not trigger a last event finish, since there is none
                        if self.last_frame_id[plane_id] > 0 and self.frame_id[plane_id] != self.last_frame_id[plane_id] + 1:
                            add_event_status(plane_id, self.event_status, DATA_ERROR)
                        self.last_frame_id[plane_id] = self.frame_id[plane_id]
                        # print 'Finsihed event', event_number[plane_id], 'for plane', plane_id
                        hit_index = finish_event(plane_id, self.hits_buffer, self.hit_buffer_index[plane_id], self.event_status[plane_id], hits, hit_index)
                    # Reset counter
                    self.hit_buffer_index[plane_id] = 0
                    self.event_status[plane_id] = 0
                    self.event_number[plane_id] += 1  # Increase event counter for this plane
                    if self.debug:
                        print("start %d" % plane_id)
                    self.word_index[plane_id] = 0
                else:
                    self.word_index[plane_id] += 1
                    if self.debug:
                        print(plane_id, self.word_index[plane_id],)
                    if self.word_index[plane_id] == 1:  # 1. word should have the header low word
                        if self.debug:
                            print("header")
                        if not is_frame_header_low(word, actual_plane):
                            add_event_status(plane_id, self.event_status, DATA_ERROR)
                    elif self.word_index[plane_id] == 2:  # 2. word should have the frame ID high word
                        if self.debug:
                            print("frame lsb")
                        self.frame_id[plane_id + 1] = get_frame_id_high(word)
                    elif self.word_index[plane_id] == 3:  # 3. word should have the frame ID low word
                        self.frame_id[actual_plane] = get_frame_id_low(word) | self.frame_id[actual_plane]
                        if plane_id == 0:
                            self.frame_id[0] = self.frame_id[actual_plane]
                    elif self.word_index[plane_id] == 4:  # 4. word should have the frame length high word
                        self.frame_length[plane_id] = get_frame_length(word)
                    elif self.word_index[plane_id] == 5:  # 5. word should have the frame length low word (=high word, one data line, the number of words is repeated 2 times)
                        if self.frame_length[plane_id] != get_frame_length(word):
                            add_event_status(plane_id, self.event_status, EVENT_INCOMPLETE)
                    elif self.word_index[plane_id] == 6 + self.frame_length[plane_id]:  # Second last word is frame tailer high word
                        if not is_frame_tailer_high(word):
                            add_event_status(plane_id, self.event_status, DATA_ERROR)
                    elif self.word_index[plane_id] == 7 + self.frame_length[plane_id]:  # First last word is frame tailer low word
                        self.frame_length[plane_id] = -1
                        self.n_words[plane_id] = 0
                        if not is_frame_tailer_low(word, actual_plane):
                            add_event_status(plane_id, self.event_status, DATA_ERROR)
                    else:  # Column / Row words
                        if self.n_words[plane_id] == 0:  # First word containing the row info and the number of data words for this row
                            if self.word_index[plane_id] == 6 + self.frame_length[plane_id] - 1:  # Always even amount of words or this fill word is used
                                add_event_status(plane_id, self.event_status, UNEVEN_EVENT)
                            else:
                                self.n_words[plane_id] = get_n_words(word)
                                self.row[plane_id] = get_row(word)
                                if self.debug:
                                    print("sts", self.n_words[plane_id], "row", self.row[plane_id])
                                if has_overflow(word):
                                    add_event_status(plane_id, self.event_status, DATA_ERROR)
                        else:
                            if self.trigger_number < 0:  # Trigger number < 0 means no trigger number
                                add_event_status(plane_id, self.event_status, NO_TRG_WORD)
                            self.n_words[plane_id] = self.n_words[plane_id] - 1  # Count down the words
                            n_hits = get_n_hits(word)
                            column = get_column(word)
                            for k in range(n_hits + 1):
                                if self.debug:
                                    print((self.event_number[plane_id], self.trigger_number, plane_id, self.frame_id[plane_id], column + k, self.row[plane_id], 0, 0))
                                out_trigger_number = 0 if self.trigger_number < 0 else self.trigger_number  # Prevent storing negative number in unsigned int
                                self.hits_buffer[plane_id, self.hit_buffer_index[plane_id]] = (self.event_number[plane_id],
                                                                                               out_trigger_number,
                                                                                               plane_id,
                                                                                               self.frame_id[plane_id],
                                                                                               column + k,
                                                                                               self.row[plane_id],
                                                                                               0,
                                                                                               0)
                                self.hit_buffer_index[plane_id] += 1
            elif is_trigger_word(word):
                self.trigger_number = get_trigger_number(word)

        return hits[:hit_index]


class DataInterpreter(object):

    def __init__(self, raw_data_file, analyzed_data_file=None, create_pdf=True, chunk_size=1000000):
        '''
        Parameters
        ----------
        raw_data_file : string or tuple, list
            A string with the raw data file name. File ending (.h5)
        analyzed_data_file : string
            The file name of the output analyzed data file. File ending (.h5)
            Does not have to be set.
        create_pdf : boolean
            Creates interpretation plots into one PDF file.
        chunk_size : integer
            How many raw data words are analyzed at once in RAM. Limited by available RAM. Faster
            interpretation for larger numbers.
        '''

        self._raw_data_file = raw_data_file

        if analyzed_data_file:
            if os.path.splitext(analyzed_data_file)[1].strip().lower() != ".h5":
                self._analyzed_data_file = os.path.splitext(analyzed_data_file)[0] + ".h5"
            else:
                self._analyzed_data_file = analyzed_data_file
        else:
            self._analyzed_data_file = os.path.splitext(self._raw_data_file)[0] + '_interpreted.h5'

        if create_pdf:
            output_pdf_filename = os.path.splitext(self._raw_data_file)[0] + ".pdf"
            logging.info('Opening output PDF file: %s', output_pdf_filename)
            self.output_pdf = PdfPages(output_pdf_filename)

        self._raw_data_interpreter = RawDataInterpreter()

        # Std. settings
        self.chunk_size = chunk_size

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if self.output_pdf:
            logging.info('Closing output PDF file: %s', str(self.output_pdf._file.fh.name))
            self.output_pdf.close()

    def interpret_word_table(self):
        with tb.open_file(self._raw_data_file, 'r') as in_file_h5:
            logging.info('Interpreting raw data file %s', self._raw_data_file)
            with tb.open_file(self._analyzed_data_file, 'w') as out_file_h5:
                description = np.zeros((1, ), dtype=hit_dtype).dtype

                hit_table = out_file_h5.create_table(out_file_h5.root,
                                                     name='Hits',
                                                     description=description,
                                                     title='hit_data',
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                                     chunkshape=(self.chunk_size / 100,))

                logging.info("Interpreting...")
                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.raw_data.shape[0], term_width=80)
                progress_bar.start()

                for word_index in range(0, in_file_h5.root.raw_data.shape[0], self.chunk_size):  # Loop over all words in the actual raw data file in chunks
                    raw_data_chunk = in_file_h5.root.raw_data.read(word_index, word_index + self.chunk_size)
                    hits = self._raw_data_interpreter.build_hits(raw_data_chunk)
                    hit_table.append(hits)
                    progress_bar.update(word_index)

                progress_bar.finish()

        # m26_decode_orig(raw_data, fout=r'C:\Users\DavidLP\git\pyBAR_mimosa26_interpreter\examples\test.txt')
#         print build_hits(raw_data)
