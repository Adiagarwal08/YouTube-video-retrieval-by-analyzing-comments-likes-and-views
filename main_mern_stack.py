import googleapiclient.discovery
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from langdetect import detect
from sklearn.preprocessing import MinMaxScaler

# Initialize YouTube API and Sentence Transformer model
API_KEY = "YOUR API KEY"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
model = SentenceTransformer('./fine_tuned_sentence_transformer')  # Stronger model for semantic similarity

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
    """Preprocess comments: clean text, remove duplicates, and filter non-informative comments."""
    processed_comments = []
    for comment in comments:
        try:
            if len(comment.split()) > 3 and detect(comment) == "en":  # Filter short and non-English comments
                processed_comments.append(clean_text(comment))
        except:
            continue  # Skip if language detection fails
    return list(set(processed_comments))  # Remove duplicates

def main(keyword, max_videos=20, max_comments=400):
    keyword_cleaned = clean_text(keyword)
    keyword_embedding = model.encode(keyword_cleaned)

    print(f"Searching for videos related to: {keyword}")
    videos = search_youtube_videos(keyword, max_videos)
    if not videos:
        print("No videos found for the given keyword.")
        return

    video_scores = []
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
        cosine_similarities = [
            util.cos_sim(keyword_embedding, model.encode(comment))[0][0].item()
            for comment in processed_comments
        ]

        # Normalize Scores and Combine
        scaler = MinMaxScaler()
        cosine_normalized = scaler.fit_transform(np.array(cosine_similarities).reshape(-1, 1)).flatten()

        avg_score = np.mean(cosine_normalized) if cosine_normalized.size > 0 else 0
        print(f"Relevance Score for video '{video_title}': {avg_score:.2f}")
        video_scores.append({"video_id": video_id, "title": video_title, "score": avg_score})

    # Sort videos by relevance score
    top_videos = sorted(video_scores, key=lambda x: x["score"], reverse=True)[:10]
    print("\nTop 10 videos based on relevance scores:")
    for video in top_videos:
        print(f"Video Title: {video['title']}, Relevance Score: {video['score']:.2f}, URL: https://www.youtube.com/watch?v={video['video_id']}")
    return top_videos

if __name__ == "__main__":
    main()
