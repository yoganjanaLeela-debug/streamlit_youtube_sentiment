# streamlit_youtube_sentiment.py

import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from googleapiclient.discovery import build

# ------------------------------------------------------
# 1. Setup YouTube Data API
# ------------------------------------------------------

API_KEY = "AIzaSyCJRdiTNS7OBYthHT9Qh6Rs96DfAB-Iz9Q"
youtube = build('youtube', 'v3', developerKey=API_KEY)

# ------------------------------------------------------
# 2. Function to Fetch Comments
# ------------------------------------------------------

def get_video_comments(video_id, max_comments=200):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            published_at = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
            comments.append({"comment": comment, "published_at": published_at})

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments)

# ------------------------------------------------------
# 3. Sentiment Analysis
# ------------------------------------------------------

def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, subjectivity, sentiment


def process_comments(video_id):
    df = get_video_comments(video_id)
    df[["polarity", "subjectivity", "sentiment"]] = df["comment"].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )
    return df

# ------------------------------------------------------
# 4. Streamlit App Interface
# ------------------------------------------------------

st.title("📺 YouTube Sentiment Analysis")

video_url = st.text_input("Enter YouTube Video URL", "https://www.youtube.com/watch?v=zEysLR8X9Ko")

if video_url:
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        st.info("Fetching comments... Please wait.")
        df = process_comments(video_id)
        st.success("Comments fetched and analyzed!")

        st.subheader("Sample Comments")
        st.dataframe(df[["comment", "sentiment"]].head())

        st.subheader("Sentiment Distribution")
        fig1 = px.histogram(df, x="sentiment", color="sentiment",
                            title="Sentiment Distribution of YouTube Comments")
        st.plotly_chart(fig1)

        st.subheader("Polarity vs Subjectivity")
        fig2 = px.scatter(df, x="polarity", y="subjectivity", color="sentiment",
                          title="Polarity vs Subjectivity")
        st.plotly_chart(fig2)

        st.subheader("Sentiment Trend Over Time")
        df["published_at"] = pd.to_datetime(df["published_at"])
        trend = df.groupby([pd.Grouper(key="published_at", freq="D"), "sentiment"]).size().reset_index(name="count")
        fig3 = px.line(trend, x="published_at", y="count", color="sentiment",
                       title="Sentiment Trend Over Time")
        st.plotly_chart(fig3)

    except Exception as e:
        st.error(f"Error: {e}")
