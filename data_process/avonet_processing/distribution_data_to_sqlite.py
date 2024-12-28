import csv
import os
import sqlite3
import sys

import shapefile

sf_path = sys.argv[1]  # path of BehrmannMeterGrid_WGS84_land.shp
csv_path = sys.argv[2]  # path of AllSpeciesBirdLifeMaps2019.csv
db_path = sys.argv[3]  # output file
csv_match_path = sys.argv[4]

sf = shapefile.Reader(sf_path)
records = sf.shapeRecords()

if os.path.exists(db_path):
    os.remove(db_path)

con = sqlite3.connect(db_path)

cur = con.cursor()

cur.execute('''CREATE TABLE places
               (worldid integer PRIMARY KEY, west real, south real, east real, north real)''')
cur.execute('''CREATE TABLE distributions
               (id integer PRIMARY KEY, species text, worldid integer)''')
cur.execute('''CREATE TABLE sp_cls_map
               (species text PRIMARY KEY, cls integer)''')

for record in records:
    box = record.shape.bbox
    cur.execute(f"INSERT INTO places VALUES ({record.record.WorldID}, {round(box[0], 2)}, {round(box[1], 2)}, {round(box[2], 2)}, {round(box[3], 2)})")

with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for index, row in enumerate(reader):
        if index == 0:
            continue
        cur.execute(f"INSERT INTO distributions VALUES ({int(row[1])}, \"{row[2]}\", {int(row[3])})")

with open(csv_match_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for index, row in enumerate(reader):
        if index == 0:
            continue
        cur.execute(f"INSERT INTO sp_cls_map VALUES (\"{row[0]}\", {int(row[1])})")


con.commit()
con.close()
