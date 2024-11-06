from supabase import create_client, Client
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Replace with your Supabase credentials
#SUPABASE_URL = ""
#SUPABASE_KEY = ""

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

# Function to convert VADER compound score (-1 to 1) to a 0-5 scale
def compound_to_rating(compound_score):
    return (compound_score + 1) * 2.5  # Scale from 0 to 5

# Analyze reviews and store sentiment scores and classifications in new columns
feedback_df["compound_score"] = feedback_df["review"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
feedback_df["sentiment"] = feedback_df["compound_score"].apply(classify_sentiment)
feedback_df["rating"] = feedback_df["compound_score"].apply(compound_to_rating)  # Convert to 0-5 rating scale

# Update feedback records in Supabase with sentiment scores and 0-5 ratings
for index, row in feedback_df.iterrows():
    supabase.table("feedback").update({
        "compound_score": row["compound_score"],
        "sentiment": row["sentiment"],
        "rating": row["rating"]
    }).eq("feedback_id", row["feedback_id"]).execute()

# Calculate average rating for each service provider (sp_id) based on new 0-5 ratings
rating_df = feedback_df.groupby("sp_id")["rating"].mean().reset_index()

# Update each service provider's rating in the 'service_provider' table
for index, row in rating_df.iterrows():
    supabase.table("service_provider").update({
        "rating": row["rating"]
    }).eq("sp_id", row["sp_id"]).execute()

print("Sentiment analysis and rating calculation job completed.")
