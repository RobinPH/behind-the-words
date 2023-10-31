import json

import pandas as pd


def JSONLReader(file_path, skip = 0, take = None):
  with open(file_path) as f:
      lines = f.read().splitlines()
      lines = lines[skip:]

      if take != None:
          lines = lines[:take]
          
  df_tmp = pd.DataFrame(lines)
  df_tmp.columns = ['json_data']

  df_tmp['json_data'].apply(json.loads)

  return pd.json_normalize(df_tmp['json_data'].apply(json.loads))