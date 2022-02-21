import datetime
import arrow
import pdb
from datetime import datetime
from collections import defaultdict
from typing import List


def is_leap_year(years: int):
    """
    if year is a leap year
    """

    assert isinstance(years, int), "Integer required."

    if ((years % 4 == 0 and years % 100 != 0) or (years % 400 == 0)):
        days_sum = 366
        return days_sum
    else:
        days_sum = 365
        return days_sum


def get_all_days_of_year(years: int, format: str = "YYYY-MM-DD") -> List[str]:
    """
    get all days of the year in string format
    """

    start_date = '%s-1-1' % years
    a = 0
    all_date_list = []
    days_sum = is_leap_year(int(years))
    while a < days_sum:
        b = arrow.get(start_date).shift(days=a).format(format)
        a += 1
        all_date_list.append(b)

    return all_date_list


def get_all_days_between(start_date, end_date, format="YYYY-MM-DD"):
    """
    get all days between starting date and ending date
    """
    start_date = arrow.get(start_date)
    end_date = arrow.get(end_date)

    res = []

    for d in range(0, (end_date-start_date).days + 1):
        new_day = start_date.shift(days=d).format(format)
        res.append(new_day)

    return res


def create_year2id(triple_time):
    """
    Preprocess for yago11k & wiki
    According to the frequency of the year in the data set, divide the time
    into several segments and create a mapping from the year segment to the index
    """
    year2id = dict()
    freq = defaultdict(int)
    year_list = []
    for k, v in triple_time.items():
        try:
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
        except:
            pdb.set_trace()

        if start.find('#') == -1 and len(start) == 4:
            year_list.append(start)
        if end.find('#') == -1 and len(end) == 4:
            year_list.append(end)
    year_list.sort()
    for year in year_list:
        freq[year] = freq[year] + 1
    year_class = []
    count = 0
    for key in sorted(freq.keys()):
        count += freq[key]
        if count > 300:
            year_class.append(key)
            count = 0
    prev_year = '0'
    i = 0
    for i, yr in enumerate(year_class):
        year2id[(prev_year, yr)] = i
        prev_year = str(int(yr) + 1)
    year2id[(prev_year, max(year_list))] = i + 1

    return year2id


def get_pretreated_data(triple_time, triples, year2id):
    """
    Preprocess for yago11k & wiki
    Sample discrete timestamps data from the original period-format data set
    """
    YEARMAX = '3000'
    YEARMIN = '-50'

    inp_idx, start_idx, end_idx = [], [], []
    for k, v in triple_time.items():
        start = v[0].split('-')[0]
        end = v[1].split('-')[0]
        if start == '####':
            start = YEARMIN
        elif start.find('#') != -1 or len(start) != 4:
            continue
        if end == '####':
            end = YEARMAX
        elif end.find('#') != -1 or len(end) != 4:
            continue
        if start > end:
            end = YEARMAX

        inp_idx.append(k)
        if start == YEARMIN:
            start_idx.append(0)
        else:
            for key, lbl in sorted(year2id.items(), key=lambda x: x[1]):
                if start >= key[0] and start <= key[1]:
                    start_idx.append(lbl)
        if end == YEARMAX:
            end_idx.append(len(year2id.keys()) - 1)
        else:
            for key, lbl in sorted(year2id.items(), key=lambda x: x[1]):
                if end >= key[0] and end <= key[1]:
                    end_idx.append(lbl)

    keep_idx = set(inp_idx)
    for i in range(len(triples) - 1, -1, -1):
        if i not in keep_idx:
            del triples[i]

    posh, rela, post = zip(*triples)
    head, rel, tail = zip(*triples)
    posh = list(posh)
    post = list(post)
    rela = list(rela)
    head = list(head)
    tail = list(tail)
    rel = list(rel)
    for i in range(len(posh)):
        if start_idx[i] < end_idx[i]:
            for j in range(start_idx[i] + 1, end_idx[i] + 1):
                head.append(posh[i])
                rel.append(rela[i])
                tail.append(post[i])
                start_idx.append(j)
    pretreated_data = []
    for i in range(len(head)):
        pretreated_data.append([head[i], rel[i], tail[i], start_idx[i]])

    return pretreated_data

