import re
from datetime import timedelta
from typing import Generator


def group_pattern(pattern):
    pattern = pattern.replace('(', '')
    pattern = pattern.replace(')', '')
    return f"({pattern})"


TIME_PATTERN = "([0-9]{2,2})\:([0-9]{2,2})\:([0-9]{2,2})\,([0-9]{3,3})"
SRT_TIME_LINE_PATTERN = f"{group_pattern(TIME_PATTERN)} --> {group_pattern(TIME_PATTERN)}"


def parse_time_str(text: str) -> timedelta:
    match = re.search(TIME_PATTERN, text)
    if not match:
        raise ValueError(f"can't parse '{text}' to timedelta.")
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))
    milliseconds = int(match.group(4))
    return timedelta(hours=hour, minutes=minute, seconds=second, milliseconds=milliseconds)


def try_parse_srt_time_line(line: str) -> [bool, timedelta, timedelta]:
    match = re.search(SRT_TIME_LINE_PATTERN, line)
    if not match:
        return False, None, None
    start_time = match.group(1)
    end_time = match.group(2)
    return True, parse_time_str(start_time), parse_time_str(end_time)


def get_srt_file_metadata_iter(srt_filepath: str) -> Generator[timedelta, timedelta, str]:
    with open(srt_filepath, "r", encoding='utf-8') as f:
        f_iter = iter(f)
        for line in f_iter:
            line = line.rstrip('\n')
            match, start_timedelta, end_timedelta = try_parse_srt_time_line(line)
            if match:
                caption = next(f_iter).rstrip('\n')
                yield start_timedelta, end_timedelta, caption


def try_parse_count_line(line: str) -> [bool, int]:
    match = re.search("^\d+$", line)
    if match:
        count = int(match.group(0))
        return [True, count]
    return [False, None]


def try_parse_time_line(line: str) -> [bool, str, str]:
    match = re.search(r'^(\d{2}:\d{2}:\d{2}\,\d+) --> (\d{2}:\d{2}:\d{2}\,\d+)$', line)
    if match:
        start = match.group(1)
        end = match.group(2)
        return [True, start, end]
    return [False, None, None]


def get_srt_file_metadata(srt_filepath: str):
    with open(srt_filepath, 'r', encoding='utf-8') as file:
        count = 0
        start_timedelta = ''
        end_timedelta = ''
        caption = ''
        read = False
        for line in iter(file):
            line = line.rstrip('\n')
            line = line.lstrip()
            match, start, end = try_parse_time_line(line)
            if match:
                start_timedelta = parse_time_str(start)
                end_timedelta = parse_time_str(end)
                caption = ''
                continue
            match, count1 = try_parse_count_line(line)
            if match:
                if read:
                    yield count, start_timedelta, end_timedelta, caption
                count = count1
                caption = ''
                continue
            read = True
            caption += line + '\r\n'
        if read:
            yield count, start_timedelta, end_timedelta, caption


def convert_timedelta_to_srt_time_format(t: timedelta) -> str:
    total_seconds = t.total_seconds()
    hour = int(total_seconds // 60 // 60)
    total_seconds -= int(hour * 60 * 60)
    minute = int(total_seconds // 60)
    total_seconds -= minute * 60
    s = str(total_seconds).split('.')
    second = int(s[0])
    milli_second = int(s[1]) if len(s) > 1 else 0
    return f"{hour:0>2d}:{minute:0>2d}:{second:0>2d},{milli_second:0>3d}"


def create_srt_timestamp_line(start_time: timedelta, end_time: timedelta) -> str:
    srt_line = f"{convert_timedelta_to_srt_time_format(start_time)}" \
               f" --> " \
               f"{convert_timedelta_to_srt_time_format(end_time)}"
    return srt_line


def create_srt_line(start_time: timedelta, end_time: timedelta, caption: str) -> str:
    srt_line = create_srt_timestamp_line(start_time, end_time)
    srt_content = f"{srt_line}\n{caption}\n"
    return srt_content
