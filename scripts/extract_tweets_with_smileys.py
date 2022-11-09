#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pyarrow as pa
import pyarrow.parquet as pq
import re
import sys

POSITIVE_SMILEYS = "😹😅😺😽😂🐱💌🌚😗💏😌💚❤😆👍💑💞😘🎉😛😚💓👌💖💛😙😁🐶😄👏😍💙🏆😊😀♡😻💘😝💗😃😜😇✌😎😋💕😉😸♥👻🌝🙂🤓🤗🤠🤣🤩🤪🥰🥳" # 💯
NEGATIVE_SMILEYS = "🤨😒🙄😬🤥🤢🤮🥵🥶😵🤯😕😟🙁😮😯😲😳😦😧😨😰😥😢😱😖😣😞😓😩😫🥱😤😡😠🤬😈👿🖕👎😔🙀😾💀😭😪💔😿🙍👺🙎" # removed for ambiguity: 😭🤔🤐✊👊🥺😷💩🆘🥴💧

def parse_tweet(obj):
  try:
    if 'extended_tweet' in obj:
      txt = obj['extended_tweet']['full_text']
    else: 
      txt = obj['text']
  except KeyError:
    txt = None
  
  idstr = obj['id_str']
  lang = obj['lang']
  return idstr,txt,lang

try:
  filename = sys.argv[1]
except IndexError:
  filename = '/nfs/scistore14/chlgrp/chl/twitter-stream/data/corona/2020/02/corona-2020-02-25.txt'

positive_smileys_re = re.compile(u'['+POSITIVE_SMILEYS+']')
negative_smileys_re = re.compile(u'['+NEGATIVE_SMILEYS+']')

positive_translation_table = str.maketrans("\n\r", "  ", POSITIVE_SMILEYS) 
negative_translation_table = str.maketrans("\n\r", "  ", NEGATIVE_SMILEYS) 

def filter(fid, min_length=10, skip_retweets=True):
  known_ids = set()
  for line in fid:
    if not line:
      continue
    try:
      obj = json.loads(line)
    except (json.decoder.JSONDecodeError, TypeError):
      #print("ERROR: entry wasn't a dictionary. skipping.", file=sys.stderr)
      continue

    try:
      if 'id_str' not in obj:
        print("ERROR: 'id' field not found in tweet", file=sys.stderr)
        continue
      if 'created_at' not in obj:
        print("ERROR: 'created_at' field not found in tweet {}".format(tweet['id']), file = sys.stderr)
        continue
      if 'retweeted_status' in obj and skip_retweets: # skip retweets
        continue 
    except TypeError:
        print("ERROR: not a dict?", line, obj, file=sys.stderr)
        continue 

    idstr, txt, lang = parse_tweet(obj)
    if not txt: # no text
      continue
    if idstr in known_ids: # duplicate
      continue
    known_ids.add(idstr)
    
    pos = re.findall(positive_smileys_re, txt)
    neg = re.findall(negative_smileys_re, txt)
    if not pos and not neg:
      continue # drop, because no smiley
    if pos and neg:
      continue # drop, because confusing
    if pos and not neg:
      txt = txt.translate(positive_translation_table)
      label = 1
    elif neg and not pos:
      txt = txt.translate(negative_translation_table)
      label = 0
    
    if len(txt)<min_length:
      continue

    smileys = ''.join(pos+neg)
    yield (idstr, txt, label, lang, smileys)

# now load data and process in batches 
# (note: batches might not be needed anymore)

bs=128

with open(filename, encoding='utf-8') as fid:
  lines = filter(fid)
  idstr,texts,labels,langs,smileys = zip(*lines)

print("Creating pyarrow table ")
data = {'id':idstr, 'text':texts, 'label':labels, 'lang':langs, 'smileys':smileys}
table = pa.Table.from_pydict(data)

try:
  output_filename = sys.argv[2]
except IndexError:
  output_filename = filename+'.parquet'

print("Writing to {}".format(output_filename))

pq.write_table(table, output_filename)
