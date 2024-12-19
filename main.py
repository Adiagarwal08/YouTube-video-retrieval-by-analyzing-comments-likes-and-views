import googleapiclient.discovery
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from langdetect import detect
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import json
import os

CACHE_FILE = "video_cache.json"  # File to store cached results

# Initialize YouTube API and Sentence Transformer model
API_KEY = "YOUR API KEY"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # Stronger model for semantic similarity

def get_video_comments(video_id, max_comments=1000):
    """Fetches all comments for a given video ID, up to max_comments."""
    comments = []
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", maxResults=100)
        response = request.execute()

        while response and len(comments) < max_comments:
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            if "nextPageToken" in response and len(comments) < max_comments:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=100,
                    pageToken=response["nextPageToken"]
                )
                response = request.execute()
            else:
                break
    except Exception as e:
        print(f"Error fetching comments for video {video_id}: {e}")
    return comments[:max_comments]

def search_youtube_videos(keyword, max_videos=20):
    """Searches YouTube for videos based on a keyword."""
    videos = []
    try:
        request = youtube.search().list(part="id,snippet", maxResults=50, q=keyword)
        response = request.execute()

        while response and len(videos) < max_videos:
            for item in response.get("items", []):
                if item["id"]["kind"] == "youtube#video":
                    video_id = item["id"]["videoId"]
                    video_title = item["snippet"]["title"]
                    
                    # Get the video duration to exclude shorts
                    video_details = youtube.videos().list(part="contentDetails", id=video_id).execute()
                    video_duration = video_details["items"][0]["contentDetails"]["duration"]

                    # Convert ISO 8601 duration to seconds
                    video_duration_seconds = parse_duration(video_duration)

                    # Exclude videos that are less than 60 seconds (Shorts)
                    if video_duration_seconds >= 120:
                        videos.append({"video_id": video_id, "title": video_title})

            if "nextPageToken" in response and len(videos) < max_videos:
                request = youtube.search().list(
                    part="id,snippet",
                    maxResults=50,
                    q=keyword,
                    pageToken=response["nextPageToken"]
                )
                response = request.execute()
            else:
                break
    except Exception as e:
        print(f"Error searching for videos with keyword {keyword}: {e}")
    return videos[:max_videos]

def parse_duration(duration):
    """Converts YouTube video duration from ISO 8601 format to seconds."""
    import re
    pattern = re.compile(r'PT(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = int(match.group(2) or 0)
        return minutes * 60 + seconds
    return 0

def clean_text(text):
    """Cleans text by removing special characters and extra spaces."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_comments(comments):
    """Preprocess comments: clean text, and filter non-informative comments."""
    processed_comments = []
    for comment in comments:
        try:
            if len(comment.split()) > 3 and detect(comment) == "en":  
                processed_comments.append(clean_text(comment))
        except:
            continue  # Skip if language detection fails
    return list(set(processed_comments))  # Remove duplicates

def draw_wordcloud(comments, video_title):
    """Generates and displays a WordCloud for the comments."""
    text = " ".join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for Video: {video_title}", fontsize=12)
    plt.show()

def analyze_sentiments(comments):
    """Performs sentiment analysis on the comments."""
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}

    for comment in comments:
        analysis = TextBlob(comment)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            sentiments["positive"] += 1
        elif polarity < 0:
            sentiments["negative"] += 1
        else:
            sentiments["neutral"] += 1

    total = sum(sentiments.values())
    if total > 0:
        print("\nSentiment Analysis:")
        for sentiment, count in sentiments.items():
            print(f"{sentiment.capitalize()} comments: {count} ({(count / total) * 100:.2f}%)")

    return sentiments  # Return the sentiment dictionary for further use

def load_cache():
    """Loads the cache file if it exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_cache(cache):
    """Saves the updated cache to the file."""
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)

def save_visualizations(videos, keyword):
    """Saves visualizations and data for the final videos in the YouTube folder."""
    base_folder = "YouTube"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)  # Create the YouTube folder if it doesn't exist

    keyword_folder = os.path.join(base_folder, keyword.replace(" ", "_"))
    if not os.path.exists(keyword_folder):
        os.makedirs(keyword_folder)

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        processed_comments = video["comments"]
        likes = video["likes"]
        views = video["views"]

        # Save WordCloud
        text = " ".join(processed_comments)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        wordcloud_path = os.path.join(keyword_folder, f"{video_id}_wordcloud.png")
        wordcloud.to_file(wordcloud_path)

        # Sentiment Analysis
        sentiments = analyze_sentiments(processed_comments)

        # Save video details
        details_file_path = os.path.join(keyword_folder, f"{video_id}_details.txt")
        with open(details_file_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n")
            f.write(f"Relevance Score: {video['score']:.2f}\n")
            f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
            f.write(f"Likes: {likes}\n")
            f.write(f"Views: {views}\n")
            f.write("\nSentiment Analysis:\n")
            total_comments = sum(sentiments.values())
            for sentiment, count in sentiments.items():
                f.write(f"{sentiment.capitalize()} Comments: {count} ({(count / total_comments) * 100:.2f}%)\n")
            f.write("\nTop Comments:\n")
            f.write("\n".join(processed_comments[:10]))  # Save top 10 comments

    print(f"Saved visualizations and data in the folder: {keyword_folder}")

def fetch_video_details(video_id):
    """Fetches likes and views for a given video ID."""
    try:
        response = youtube.videos().list(part="statistics", id=video_id).execute()
        stats = response["items"][0]["statistics"]
        likes = int(stats.get("likeCount", 0))  # Likes may not always be available
        views = int(stats.get("viewCount", 0))
        return likes, views
    except Exception as e:
        print(f"Error fetching details for video {video_id}: {e}")
        return 0, 0  # Default to 0 if there's an error
    
def calculate_relevance_score(cosine_scores, likes, views):
    """Calculates the weighted relevance score."""
    scaler = MinMaxScaler()

    # Normalize cosine similarity scores
    cosine_normalized = scaler.fit_transform(np.array(cosine_scores).reshape(-1, 1)).flatten()

    # Normalize likes and views
    likes_normalized = scaler.fit_transform(np.array(likes).reshape(-1, 1)).flatten() if likes else [0]
    views_normalized = scaler.fit_transform(np.array(views).reshape(-1, 1)).flatten() if views else [0]

    # Weighted relevance score
    w1, w2, w3 = 0.5, 0.3, 0.2
    relevance_scores = [
        w1 * cosine + w2 * like + w3 * view
        for cosine, like, view in zip(cosine_normalized, likes_normalized, views_normalized)
    ]

    return relevance_scores



def main():
    cache = load_cache()

    keyword_user = input("Enter the topic you want to search for: ").strip().lower()
    max_videos = int(input("Enter the number of videos to process (e.g., 20): "))
    max_comments = int(input("Enter the maximum number of comments per video (e.g., 1000): "))

    if keyword_user in cache:
        print(f"\nResults for '{keyword_user}' loaded from cache.")
        top_videos = cache[keyword_user]
    else:
        print(f"\nSearching for videos related to: {keyword_user}")
        videos = search_youtube_videos(keyword_user, max_videos)
        if not videos:
            print("No videos found for the given keyword.")
            return

        video_scores = []
        all_likes = []
        all_views = []
        all_cosine_scores = []
        for video in videos:
            video_id = video["video_id"]
            video_title = video["title"]

            print(f"\nFetching comments for video: {video_title} (ID: {video_id})")
            comments = get_video_comments(video_id, max_comments)
            processed_comments = process_comments(comments)
            if not processed_comments:
                print("No meaningful comments found for this video.")
                continue

            # Compute Cosine Similarities between comments and the keyword
            keyword_cleaned = clean_text(keyword_user)
            keyword_embedding = model.encode(keyword_cleaned)
            cosine_similarities = [
                util.cos_sim(keyword_embedding, model.encode(comment))[0][0].item()
                for comment in processed_comments
            ]

            avg_cosine_score = np.mean(cosine_similarities) if cosine_similarities else 0

            # Fetch likes and views
            likes, views = fetch_video_details(video_id)
            all_likes.append(likes)
            all_views.append(views)
            all_cosine_scores.append(avg_cosine_score)

            video_scores.append({
                "video_id": video_id,
                "title": video_title,
                "comments": processed_comments,
                "likes": likes,
                "views": views
            })

    # Calculate relevance scores using comments, likes, and views
        relevance_scores = calculate_relevance_score(all_cosine_scores, all_likes, all_views)

        # Update videos with calculated relevance scores
        for i, video in enumerate(video_scores):
            video["score"] = relevance_scores[i]

        # Sort videos by relevance score
        top_videos = sorted(video_scores, key=lambda x: x["score"], reverse=True)[:10]
        cache[keyword_user] = top_videos
        save_cache(cache)

    print("\nTop 10 videos based on relevance scores:")
    for video in top_videos:
        print(f"Video Title: {video['title']}, Relevance Score: {video['score']:.2f}, URL: https://www.youtube.com/watch?v={video['video_id']}")

    # Save visualizations and data for the final videos
    save_visualizations(top_videos, keyword_user)


if __name__ == "__main__":
    main()



   

