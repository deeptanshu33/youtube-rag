import json
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from yt_rag import YTRag


def extract_video_id(target: str) -> str:
    """Extract YouTube video ID from a full/shortened URL or return the raw ID."""
    target = target.strip()
    if not target:
        return ""
    if "v=" in target:
        query_part = target.split("v=", 1)[1]
        return query_part.split("&", 1)[0].split("#", 1)[0]
    if "youtu.be/" in target:
        path_part = target.split("youtu.be/", 1)[1]
        return path_part.split("?", 1)[0].split("#", 1)[0]
    if "embed/" in target:
        path_part = target.split("embed/", 1)[1]
        return path_part.split("?", 1)[0].split("#", 1)[0]
    return target

def get_youtube_title(video_url):
        # Set up the oEmbed query parameters
        params = {"format": "json", "url": video_url}
        oembed_url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(params)
        
        try:
            # Fetch data from the oEmbed endpoint
            with urllib.request.urlopen(oembed_url) as response:
                data = json.loads(response.read().decode())
                return data.get("title")
        except Exception as e:
            return f"Error retrieving title: {e}"


def main():
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    yt_rag = YTRag(llm)

    print("=" * 60)
    print(" YouTubeRAG Assistant")
    print(" Commands:")
    print("   set video <id or URL> : Index a new YouTube video")
    print("   quit / exit          : Exit the program")
    print("=" * 60)

    while True:
        try:
            command = input("ask a question " + f"({yt_rag.curr_video_title if len(yt_rag.curr_video_title)>0 else "no video selected"}) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not command:
            continue

        cmd_lower = command.lower()
        if cmd_lower in ("quit", "exit"):
            print("Exiting. Goodbye!")
            break

        if cmd_lower == "set video":
            target = input("Enter YouTube Video ID or URL > ").strip()
            video_id = extract_video_id(target)
            video_title = get_youtube_title(target)
            yt_rag.curr_video_title = video_title
            if not video_id:
                print("Error: No video ID provided.")
                continue
            try:
                print(f"Indexing video '{yt_rag.curr_video_title}'...")
                yt_rag.choose_video(video_id)
                print(f"Video '{yt_rag.curr_video_title}' is ready.")
            except Exception as e:
                print(f"Error loading video '{video_id}': {e}")
        elif cmd_lower.startswith("set video "):
            target = command[len("set video "):].strip()
            video_id = extract_video_id(target)
            video_title = get_youtube_title(target)
            yt_rag.curr_video_title = video_title
            if not video_id:
                print("Error: No video ID provided.")
                continue
            try:
                print(f"Indexing video '{yt_rag.curr_video_title}'...")
                yt_rag.choose_video(video_id)
                print(f"Video '{yt_rag.curr_video_title}' is ready.")
            except Exception as e:
                print(f"Error loading video '{video_id}': {e}")
        else:
            try:
                answer = yt_rag.run(command)
                print("-"*60)
                print("\n")
                print(answer)
                print("\n")
                print("-"*60)
            except Exception as e:
                print(f"Error generating answer: {e}")


if __name__ == "__main__":
    main()