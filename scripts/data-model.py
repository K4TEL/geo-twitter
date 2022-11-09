import pandas as pd   
import sys
from shapely.geometry import Point, LineString
import geopandas as gpd
from geopandas import GeoDataFrame
import spacy
import os.path
from os import path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import geopy.distance
import matplotlib.pyplot as plt

train_set = "geo.jsonl"
temp = "nlp.jsonl"
test = "test.jsonl"
result = "result-small.jsonl"

empty = "empty-twitter-example-big.jsonl"

nlp_model = "en_core_web_sm"
world_model = "naturalearth_lowres"

def read_json(file):      
    print("=== Reading data from:", file)
    try:
        df = pd.read_json(path_or_buf=file, lines=True)
        #print(df.head(5))
        print(df)
    except:
        print("=== Can't read data from:", file)
    
    return df

def coords_on_map(df, long="long", lat="lat", clr="red"):
    geometry = [Point(xy) for xy in zip(df[long], df[lat])]
    gdf = GeoDataFrame(df, geometry=geometry)   
    world = gpd.read_file(gpd.datasets.get_path(world_model))
    
    gdf.plot(ax=world.plot(color='white', 
                       edgecolor='black', 
                       figsize=(20, 16)), 
    marker='o', 
    color=clr, 
    markersize=5);
    
def lines_on_map(df, x1="long", y1="lat", x2="long_gc", y2="lat_gc", clr="red"):
    world_ax = gpd.read_file(gpd.datasets
                             .get_path(world_model)).plot(
                                     color='white', 
                                     edgecolor='yellow', 
                                     figsize=(20, 16))
    
    geometry = [LineString([[x1,y1], [x2,y2]]) for 
                x1,y1,x2,y2 in zip(df[x1], df[y1], df[x2], df[y2])]
    gdf = GeoDataFrame(df, geometry=geometry)   
    gdf.plot(ax=world_ax, marker=None, color="black", markersize=1);
    
    geometry = [Point(xy) for xy in zip(df[x2], df[y2])]
    gdf = GeoDataFrame(df, geometry=geometry)   
    gdf.plot(ax=world_ax, marker='o', color=clr, markersize=5, zorder=3)
    
    geometry = [Point(xy) for xy in zip(df[x1], df[y1])]
    gdf = GeoDataFrame(df, geometry=geometry)   
    gdf.plot(ax=world_ax, marker='o', color="green", markersize=10, zorder=2)


    
def spacy_ner(df):
    if not path.exists("model"):
        print("=== Local folder of NLP model not found. Downloading")
        spacy.cli.download(nlp_model)
        nlp = spacy.load('model') 
        nlp.to_disk('model')
    else:
        nlp = spacy.load('model') 
    
    df['gpe'] = None
    df['loc'] = None
    df['match_geo'] = False
    
    print("=== NER Processing text of tweets to get GPE and LOC")
    
    for ind in df.index:
        doc = nlp(df['text'][ind])
        for ent in doc.ents:
            if (ent.label_ == 'GPE'):
                gpe = ent.text
                df['gpe'][ind] = gpe
                if gpe:
                    df['match_geo'][ind] = str(df['place'][ind]).find(gpe) != -1
                    
            if (ent.label_ == 'LOC'):
                loc = ent.text
                df['loc'][ind] = loc
                if loc:
                    df['match_geo'][ind] = str(df['place'][ind]).find(loc) != -1
                    
    print("===", str(round(sum(df["match_geo"])/len(df["id"])*100, 2)), 
          "% of ALL processed data match decoded Geo/Loc")
                    
    return df

def geocoding(df, city_country=True, coords=False):
    geolocator = Nominatim(user_agent="geoapiExercises")
    
    print("OpenStreetMap Geolocator is set up")
    
    def city_country(row):
        try:
            coord = f"{row['lat']}, {row['long']}"
            location = geolocator.reverse(coord, exactly_one=True, timeout=None)
            address = location.raw['address']
            city = address.get('city', '')
            country = address.get('country', '')
            row['city'] = city
            row['country'] = country
        except GeocoderTimedOut as e:
            print("Error: geocode failed on input %s with coordinates %s"%(f"{row['lat']}, {row['long']}", e.message))
        
        if row['city'] is None or row['country'] is None:
            print(row["id"], "not found for Coords:", row['long'], row["lat"], 
                  "==> City:", row['city'], "Country:", row["country"])
        
        return row
    
    def coords(row):
        try:
            location = geolocator.geocode(row['gpe'], timeout=None)
            if location:
                row['lat_gc'] = location.latitude
                row['long_gc'] = location.longitude
        except GeocoderTimedOut as e:
            print("Error: geocode failed on input %s with GPE %s"%(row['gpe'], e.message))
            
        if row['long_gc'] is None:
            print(row["id"], "not found for GPE:", row["gpe"], 
                  "==> Long:", row['long_gc'], "Lat:", row["lat_gc"])
        
        return row
    
    if coords:
        print("=== Geocoding GPE to long, lat in progress")
        
        df["long_gc"] = None
        df["lat_gc"] = None
        
        df = df.apply(coords, axis=1)
        
        print("===", str(round(sum(df["long_gc"].notna())/len(df["id"])*100, 2)), 
              "% of processed data attributed coordinates")
        
    if city_country:
        print("=== Reverse geocoding long, lat to city, country in progress")
        
        df["city"] = None
        df["country"] = None
        
        df = df.apply(city_country, axis=1)
        
        print("===", str(round(sum(df["city"].notna())/len(df["id"])*100, 2)), 
              "% of processed data attributed city")
        print("===", str(round(sum(df["country"].notna())/len(df["id"])*100, 2)), 
              "% of processed data attributed country")
    
    return df

def save_df(file, df):
    with open(file, "w") as f:
        df.to_json(f, orient='records', lines=True)
    print("=== Data saved to file", file)
    
def read_args(train=False, empty=False):
    try: # 1 arg - data imput 
      file = sys.argv[1]
    except IndexError:
      file = train_set
      
    if train:
      train_data = read_json(file)
      #coords_on_map(train_data)
    else:
      train_data = None
      
    try: # 2 arg - data imput withut geo
      file = sys.argv[2]
      
    except IndexError:
      file = empty
      
    if empty:
      empty_data = read_json(file)
    else:
      empty_data = None
      
    return train_data, empty_data
    
def main():
    train_data, empty_data = read_args(True, False) # load train and empty set
    
    # NER layer
    geo = spacy_ner(train_data)
    geo = geo[geo['gpe'].notna()]
    #print(geo.head(5))
    #print(geo.info())
    print("===", str(round(sum(geo["match_geo"])/len(geo["id"])*100, 2)), 
          "% of SUCCESSFULLY processed data match decoded Geo/Loc")
    save_df(temp, geo)
    
    # Geocoding layer
    geo = read_json(temp)
    df = geocoding(geo, True, True)
    save_df(temp, df)
    
    # Coordinates processing
    df = read_json(temp)
    df = df[df['long_gc'].notna()]
    
    def distance(row):
        dist = geopy.distance.geodesic(
                    (row['lat'],row['long']),(row['lat_gc'],row['long_gc']))
        row["dist"] = dist.meters
        return row
    
    df = df.apply(distance, axis=1)
    print(df.info())
    save_df(result, df)
    
    x=df.loc[df['match_geo']==1, 'dist']
    y=df.loc[df['match_geo']==0, 'dist']
    
    bins=list(range(100))
    
    plt.figure(figsize=(18,8))
    plt.title("Distance of coords mismatch by match in 'place' column")
    plt.hist(x, bins, alpha=0.5, label='true')
    plt.hist(y, bins, alpha=0.5, label='false')
    plt.legend(loc='upper right')
    plt.show()
    
    lines_on_map(df)
    
if __name__ == "__main__":
    main()
