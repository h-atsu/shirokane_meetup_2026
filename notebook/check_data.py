# %%
import plotly.express as px
import polars as pl
from IPython.display import display

from shirokane_meetup_2026.consts import ROOT

# %%
INSTANCE_PATH = ROOT / "data" / "users_10000_contents_100_seed_0"


# %%
df_user = pl.read_parquet(INSTANCE_PATH / "users.parquet")
df_contents = pl.read_parquet(INSTANCE_PATH / "contents.parquet")
df_user_contents = pl.read_parquet(INSTANCE_PATH / "user_contents.parquet")

print(df_user.shape)
display(df_user.head())
print(df_contents.shape)
display(df_contents.head())
print(df_user_contents.shape)
display(df_user_contents.head())

# %%
df = (
    df_user_contents.group_by("content_id")
    .agg(pl.col("score").sum())
    .sort("score", descending=True)
)
px.bar(df, y="score", title="Total score per content")

# %%
