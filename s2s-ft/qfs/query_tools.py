# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from os.path import join
import json

"""
    This file is copied and revised from the shiftsum codebase.

    Original file path: shiftsum/src/querysum/tools/query_tools.py
"""


def get_annual_masked_query(year, query_type):
    proj_root = Path('/disk/nfs/ostrom/s1617290/shiftsum')
    dp_data = proj_root / 'data'
    dp_masked_query = dp_data / 'masked_query'
    NARR = 'narr'
    TITLE = 'title'
    QUERY = 'query'

    fp = dp_masked_query / f'{year}.json'
    annual_dict = dict()
    with open(fp) as f:
        for line in f:
            json_obj = json.loads(line.rstrip('\n'))
            cid = json_obj['cid']

            masked_query = json_obj['masked_query']

            if query_type == TITLE:
                masked_query = masked_query[0]
            elif query_type == NARR:
                masked_query = ' '.join(masked_query[1:])
            elif query_type == QUERY:
                masked_query = ' '.join(masked_query)
            else:
                raise ValueError(f'Invalid query_type: {query_type}')

            print(f'masked_query: {masked_query}')
            annual_dict[cid] = masked_query
    return annual_dict


def get_cid2masked_query(query_type):
    years = ['2005', '2006', '2007']
    query_dict = dict()
    for year in years:
        annual_dict = get_annual_masked_query(year, query_type=query_type)
        query_dict = {
            **annual_dict,
            **query_dict,
        }
    return query_dict
