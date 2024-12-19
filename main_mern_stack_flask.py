from flask import Flask, render_template, request
import model4  # Ensure `model4` is correctly implemented
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from wordcloud import WordCloud
import os
from textblob import TextBlob

app = Flask(__name__)

# Directory for storing generated plots
if not os.path.exists("static/plots"):
    os.makedirs("static/plots")

# Store the video data in the app's config for persistence between routes
app.config["VIDEOS"] = []

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/search_topic", methods=["GET", "POST"])
def search_topic():
    if request.method == "POST":
        topic = request.form["topic"]

        try:
            # Fetch videos using the `main` function from `model4`
            videos = model4.main(topic)
            if not videos:  # Check if no videos are found
                return render_template("result.html", topic=topic, videos=[], error="No videos found for the given topic.")
        except Exception as e:
            return render_template("error.html", message=f"Error fetching videos: {str(e)}")

        # Save the videos in the app's config
        app.config["VIDEOS"] = videos

        # Prepare data for rendering
        video_data = []
        for video in videos:
            video_id, video_title, relevance_score = video["video_id"], video["title"], video["score"]
            img_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_data.append({
                "id": video_id,
                "title": video_title,
                "img_url": img_url,
                "video_url": video_url,
                "score": relevance_score
            })

        return render_template("result.html", topic=topic, videos=video_data)

    return render_template("search_topic.html")

@app.route("/visualize/<video_id>")
def visualize(video_id):
    # Retrieve video data from the app's config
    videos = app.config.get("VIDEOS", [])
    video_details = next((video for video in videos if video["video_id"] == video_id), None)

    if not video_details:
        return render_template("error.html", message="Video not found.")

    try:
        # Fetch video comments and perform analysis
        comments = model4.get_video_comments(video_id)
        if not comments:  # Handle case where no comments are found
            return render_template("error.html", message=f"No comments found for video ID {video_id}")

        processed_comments = model4.process_comments(comments)

        # Visualization 1: Sentiment Analysis
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for comment in processed_comments:
            sentiment = TextBlob(comment).sentiment.polarity
            if sentiment > 0:
                sentiments["positive"] += 1
            elif sentiment < 0:
                sentiments["negative"] += 1
            else:
                sentiments["neutral"] += 1
        sentiment_chart_path = save_sentiment_pie_chart(video_id, sentiments)

        # Visualization 2: Comment Word Cloud
        wordcloud_path = save_wordcloud_image(video_id, processed_comments)

       

        # Pass paths to render in HTML
        plots = {
            "sentiment_pie": sentiment_chart_path,
            "word_cloud": wordcloud_path
        }

        return render_template("visualization.html", video=video_details, plots=plots)
    except Exception as e:
        return render_template("error.html", message=f"Error processing video {video_id}: {str(e)}")

def save_sentiment_pie_chart(video_id, sentiments):
    labels = list(sentiments.keys())
    sizes = list(sentiments.values())
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["green", "yellow", "red"])
    plt.title("Sentiment Distribution")
    sentiment_chart_path = f"{video_id}_sentiment_analysis.png"
    plt.savefig(os.path.join('static/plots',sentiment_chart_path))
    plt.close()
    return sentiment_chart_path

def save_wordcloud_image(video_id, comments):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(comments))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Comment Word Cloud")
    wordcloud_path = f"{video_id}_wordcloud.png"
    plt.savefig(os.path.join('static/plots',wordcloud_path))
    plt.close()
    return wordcloud_path

if __name__ == "__main__":
    app.run(debug=True)
