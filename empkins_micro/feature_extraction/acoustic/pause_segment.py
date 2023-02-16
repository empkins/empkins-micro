import os
import numpy as np
import pandas as pd
import webrtcvad
from pydub import AudioSegment
import sys
import contextlib
import wave
import collections
from empkins_micro.feature_extraction.acoustic.helper import get_length


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset: offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_get_segment_times(
        sample_rate, frame_duration_ms, padding_duration_ms, vad, frames
):
    """Filters out non-voiced audio frames.
    BT: based on vad_collector, but returns start and end times for voiced segs
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: lists of start and end segments
    """

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    start_times = []
    end_times = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                start_times.append(ring_buffer[0][0].timestamp)  # BT
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                end_times.append(ring_buffer[0][0].timestamp + frame.duration)  # BT
                triggered = False

    if triggered:  # BT if were in triggered state at end of signal, set output time
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        if len(ring_buffer) > 0:
            end_times.append(ring_buffer[0][0].timestamp)  # BT
        else:
            # only get here in very rare case that we triggered on 2nd-to-last frame
            end_times.append(frame.timestamp + frame.duration)
    # sys.stdout.write('\n')

    return (start_times, end_times)


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write("1" if is_speech else "0")
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write("+(%s)" % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
                triggered = False
                yield b"".join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:  # BT if were in triggered state at end of signal, set output time
        sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
    sys.stdout.write("\n")
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b"".join([f.bytes for f in voiced_frames])


def filter_seg_times(seg_starts, seg_ends, pad_at_start=0.5, len_to_keep=2.5):
    """
    do some filtering on the segments found to select part for analysis
    rule: find the first segment that is at least (pad_at_start+len_to_keep sec long.
    Discard the firstpad_at_start sec, keep the next len_to_keep sec
    if no such segments, then return empty list
    returns sel_start, sel_end, sel_end_longer
    """
    sel_start = []
    sel_end = []
    sel_end_longer = []

    not_found = True
    for iseg in range(len(seg_starts)):
        seg_dur = seg_ends[iseg] - seg_starts[iseg]
        if not_found & (seg_dur > (pad_at_start + len_to_keep)):
            t_start = seg_starts[iseg] + pad_at_start
            sel_start.append(t_start)
            sel_end.append(t_start + len_to_keep)
            sel_end_longer.append(
                max(t_start + len_to_keep, seg_ends[iseg] - pad_at_start)
            )
            not_found = False

    return sel_start, sel_end, sel_end_longer


def get_timing_cues(seg_starts_sec, seg_ends_sec):
    """
    Get timing cues from segmented speech
    Args:
        seg_starts_sec: Audio segment start time in seconds
        seg_ends_sec: Audio segment end time in seconds
    Returns:
        Dictionary with pause features
    """
    total_time = seg_ends_sec[-1] - seg_starts_sec[0]
    speaking_time = np.sum(np.asarray(seg_ends_sec) - np.asarray(seg_starts_sec))
    num_pauses = len(seg_starts_sec) - 1
    pause_len = np.zeros(num_pauses)

    for p in range(num_pauses):
        pause_len[p] = seg_starts_sec[p + 1] - seg_ends_sec[p]

    if len(pause_len) > 0:
        pause_time = np.sum(pause_len)

    else:
        pause_time = 0

    pause_frac = pause_time / total_time

    timing_dict = {
        'aco_totaltime': total_time,
        'aco_speakingtime': speaking_time,
        'aco_numpauses': num_pauses,
        'aco_pausetime': pause_time,
        'aco_pausefrac': pause_frac,
        'error': "PASS"
    }
    return timing_dict


def process_silence(audio_file):
    """
    Returns dataframe for pause between words using voice activity detection
    Args:
        audio_file: Audio file location
    Returns:
        Dataframe value
    """
    feat_dict_list = []
    y, sr = read_wave(str(audio_file))

    # 3 is most aggressive (splits most), 0 least (better for low snr)
    aggressiveness = 3
    frame_dur_ms = 20

    # pause segment(long & short pad)
    long_pad_around_voice_ms = 200
    short_pad_around_voice_ms = 100

    if len(y) > 0:
        vad = webrtcvad.Vad(aggressiveness)

        frames = frame_generator(frame_dur_ms, y, sr)
        frames = list(frames)

        # longer pad time screens out little blips, but misses short silences
        long_seg_starts, long_seg_ends = vad_get_segment_times(
            sr, frame_dur_ms, long_pad_around_voice_ms, vad, frames
        )

        # Logic to handle blank audio file
        if len(long_seg_starts) == 0 or len(long_seg_ends) == 0:
            return empty_pause_segment("blank audio file")

        t_start = long_seg_starts[0]
        t_end = long_seg_ends[-1]
        # shorter pad time captures short silences (but misfires on little blips)
        short_seg_starts, short_seg_ends = vad_get_segment_times(
            sr, frame_dur_ms, short_pad_around_voice_ms, vad, frames
        )

        seg_starts = []
        seg_ends = []
        for k in range(
                len(short_seg_starts)
        ):  # logic to clean up some typical misfires
            if (short_seg_starts[k] >= t_start) and (short_seg_starts[k] <= t_end):
                seg_starts.append(short_seg_starts[k])
                seg_ends.append(short_seg_ends[k])
        if len(seg_starts) == 0 or len(seg_ends) == 0:
            return empty_pause_segment("webrtcvad returns no segment")

        timing_dict = get_timing_cues(seg_starts, seg_ends)
        feat_dict_list.append(timing_dict)

    else:
        return empty_pause_segment("audio segment is empty")

    df = pd.DataFrame(feat_dict_list)
    return df


def empty_pause_segment(error_text):
    data = {
        'aco_totaltime': [np.nan],
        'aco_speakingtime': [np.nan],
        'aco_numpauses': [np.nan],
        'aco_pausetime': [np.nan],
        'aco_pausefrac': [np.nan],
        'error': [error_text]
    }
    return pd.DataFrame.from_dict(data)


def calc_pause_segment(audio_file, mono_wav):
    """
    Processing all patient's for getting Pause Segment
    ---------------
    ---------------
    Args:
        video_uri: video path; r_config: raw variable config object
        out_dir: (str) Output directory for processed output
    """
    try:
        audio_duration = get_length(audio_file)

        if float(audio_duration) < 0.064:
            return empty_pause_segment("audio duration less than 0.064 seconds")

        # Converting stereo sound to mono-lD
        sound_mono = AudioSegment.from_wav(audio_file)
        sound_mono = sound_mono.set_channels(1)
        sound_mono = sound_mono.set_frame_rate(48000)
        sound_mono.export(mono_wav, format="wav")

        df_pause_seg = process_silence(mono_wav)
        os.remove(mono_wav)  # removing mono wav file

        return df_pause_seg

    except Exception as e:
        return empty_pause_segment(f"failed to process audio file: {e}")
