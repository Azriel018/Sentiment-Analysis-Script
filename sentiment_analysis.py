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
response = supabase.table("feedback").select("feedback_id, pet_owner_id, sp_id, review").execute()
feedback_df = pd.DataFrame(response.data)

# Function to classify sentiment based on VADER's compound score
def classify_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Analyze reviews and store sentiment scores and classifications in new columns
feedback_df["compound_score"] = feedback_df["review"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
feedback_df["sentiment"] = feedback_df["compound_score"].apply(classify_sentiment)

# Update feedback records in Supabase with sentiment scores
for index, row in feedback_df.iterrows():
    supabase.table("feedback").update({
        "compound_score": row["compound_score"],
        "sentiment": row["sentiment"]
    }).eq("feedback_id", row["feedback_id"]).execute()

# Calculate the average compound score for each service provider (sp_id)
compound_score_df = feedback_df.groupby("sp_id")["compound_score"].mean().reset_index()

# Update each service provider's average compound score in the 'rating' column of the 'service_provider' table
for index, row in compound_score_df.iterrows():
    supabase.table("service_provider").update({
        "rating": row["compound_score"]
    }).eq("sp_id", row["sp_id"]).execute()

print("Sentiment analysis and rating update job completed.")
