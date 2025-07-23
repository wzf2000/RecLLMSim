import pandas as pd

from read_profile import get_sim_profile, get_human_profile
from utils import HUMAN_DIR, HUMAN_DIR_V2

df1 = get_sim_profile()
df2 = get_human_profile(HUMAN_DIR)
df3 = get_human_profile(HUMAN_DIR_V2)
df2 = pd.concat([df2, df3], ignore_index=True)

print(df2['gender'].value_counts())
print(df2['occupation'].value_counts())
print(df2['age'].value_counts())
print(df2['background'].value_counts())
