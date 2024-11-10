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

def fetch_new_feedback():
    """Fetch feedback records without a compound score."""
    response = supabase.table("feedback").select("feedback_id, review").is_("compound_score", None).execute()
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()

def calculate_and_update_compound_scores(feedback_df):
    """Calculate compound scores and update the feedback table."""
    feedback_df["compound_score"] = feedback_df["review"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    for index, row in feedback_df.iterrows():
        supabase.table("feedback").update({
            "compound_score": row["compound_score"]
        }).eq("feedback_id", row["feedback_id"]).execute()

def main():
    # Fetch new feedback that requires a compound score calculation
    feedback_df = fetch_new_feedback()
    
    if not feedback_df.empty:
        print("Calculating compound scores for new feedback entries...")
        calculate_and_update_compound_scores(feedback_df)
        print("Compound score calculation and update job completed.")
    else:
        print("No new feedback entries to process.")

if __name__ == "__main__":
    main()
