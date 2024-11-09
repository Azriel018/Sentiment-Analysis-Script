import os
from supabase import create_client, Client
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Fetch Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Check if credentials are properly fetched
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are not set in environment variables")

# Create Supabase client instance
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Fetch data from the 'feedback' table
response = supabase.table("feedback").select("feedback_id, review").execute()
feedback_df = pd.DataFrame(response.data)

# Analyze each review to get the compound score and update the feedback table
feedback_df["compound_score"] = feedback_df["review"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

# Update feedback records in Supabase with only the compound score
for index, row in feedback_df.iterrows():
    supabase.table("feedback").update({
        "compound_score": row["compound_score"]
    }).eq("feedback_id", row["feedback_id"]).execute()

print("Compound score calculation and update job completed.")
