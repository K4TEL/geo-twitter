import json
import sys
import pandas as pd
import os, glob

input_file = '/run/user/1005618/gvfs/smb-share:server=archive3.ist.local,share=group/chlgrp/twitter-collection-2022/twitter-2022-01-25.txt'
output_file = "datasets/"

country_codes = ["CA", "GB", "FR"]

def parse_geo(obj):
    print(obj['coordinates']["coordinates"])
    coord_long = obj['coordinates']["coordinates"][0]
    coord_lat = obj['coordinates']["coordinates"][1]
    return coord_long,coord_lat

def parse_user(obj):
    location = obj['user']['location'] if obj["user"]["location"] else ""
    username = obj['user']['name'] if obj["user"]["name"] else ""
    screen = obj['user']['screen_name'] if obj["user"]["screen_name"] else ""
    description = obj['user']['description'] if obj["user"]["description"] else ""
    user = f"{username} {screen} {description} {location}"
    return user

def parse_place(obj):
    full = obj['place']['full_name'] if obj['place']['full_name'] else ""
    country = obj['place']['country'] if obj['place']['country'] else ""
    code = obj['place']['country_code'] if obj['place']['full_name'] else ""
    name = obj['place']['name'] if obj['place']['country_code'] else ""
    type = obj['place']['place_type'] if obj['place']['place_type'] else ""
    place = f"{country} {type} {name} {full} {code}"
    return place, code

def parse_tweet(obj):
    text = obj['text']
    time = obj['created_at']
    lang = obj['lang'] if obj['lang'] else ""
    return text, time, lang

def parse_train(fid):
    known_ids = set()

    for line in fid:
        if not line:
          continue
        try:
          obj = json.loads(line)
        except (json.decoder.JSONDecodeError, TypeError):
          print("ERROR: entry wasn't a dictionary. skipping.", file=sys.stderr)
          continue

        try:
          if 'id_str' not in obj:
            print("ERROR: 'id' field not found in tweet", file=sys.stderr)
            continue
          if 'place' not in obj:
            print("ERROR: 'place' field not found in tweet", file=sys.stderr)
            continue
          if 'user' not in obj:
            print("ERROR: 'user' field not found in tweet", file=sys.stderr)
            continue
          if 'coordinates' not in obj:
            print("ERROR: 'coordinates' field not found in tweet", file=sys.stderr)
            continue
          if 'created_at' not in obj:
            print("ERROR: 'created_at' field not found in tweet {}".format(tweet['id']), file = sys.stderr)
            continue

        except TypeError:
            print("ERROR: not a dict?", line, obj, file=sys.stderr)
            continue

        if not obj["coordinates"]:
            continue
        if not obj["coordinates"]["coordinates"]:
            continue
        if not obj["place"]:
            continue
        # if obj["place"]["country_code"] not in country_codes:
        #     continue
        if not obj["user"]:
            continue

        if obj['id_str'] in known_ids: # duplicate
          continue

        text, time, lang = parse_tweet(obj)
        long, lat = parse_geo(obj)
        place, code = parse_place(obj)
        user = parse_user(obj)
        known_ids.add(id)

        yield (long, lat, text, time, lang, code, place, user)

def read_train(filename):
    print("Reading data from:", filename)
    with open(filename, encoding='utf-8') as fid:
      lines = parse_train(fid)
      longs, lats, texts, times, langs, codes, places, users = zip(*lines)

    data_geo = {
        'longitude':longs,
        'latitude':lats,
        'time':times,
        'texts':texts,
        'lang':langs,
        'code':codes,
        'place':places,
        'user':users
    }

    print("Training set of ===", len(longs), "=== samples is collected")

    return pd.DataFrame(data_geo)

def write_records(file, df):
    with open(file, "w") as f:
        df.to_json(f, orient='records', lines=True)
    print("Data written to file:", file)

def combine(file):
    os.chdir(os.path.dirname(__file__) + r"/filtered_json")

    extension = 'txt'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    combined_txt = pd.concat([pd.read_json(path_or_buf=f, lines=True) for f in all_filenames ])

    #os.chdir(os.path.dirname(__file__))

    with open(file, "w") as f:
        combined_txt.to_json(file, orient='records', lines=True)
    print(f"Data from {len(all_filenames)} files written to common dataset: {file}")

def main():
    try: # 1 arg - data input
        filename = sys.argv[1]
    except IndexError:
        filename = input_file
    print(f"Input file: {filename}")

    try: # 2 arg - filtered data output
      output_filename = sys.argv[2]
    except IndexError:
      head, tail = os.path.split(filename)
      output_filename = output_file + tail
      open(output_filename, 'w').close()
    print(f"Output file: {output_filename}")

    # df_geo = read_train(filename)
    # write_records(output_filename, df_geo)

    try:
        df_geo = read_train(filename)
        write_records(output_filename, df_geo)
    except Exception as e:
        print("Couldn't form dataset:", e)

    #combine("ca-twitter-2022.jsonl")

if __name__ == "__main__":
    main()
