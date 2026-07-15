import pandas as pd
import json

df = pd.read_excel('clients.xlsx')
print(json.dumps(df.to_dict(orient='records'), indent=2))
