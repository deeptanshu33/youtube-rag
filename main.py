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


def main():
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    yt_rag = YTRag(llm)

    print("=" * 60)
    print(" YouTube Transcript RAG Assistant")
    print(" Commands:")
    print("   set video <id or URL> : Index a new YouTube video")
    print("   quit / exit          : Exit the program")
    print("=" * 60)

    while True:
        try:
            command = input("ask a question > ").strip()
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
            if not video_id:
                print("Error: No video ID provided.")
                continue
            try:
                print(f"Indexing video '{video_id}'...")
                yt_rag.choose_video(video_id)
                print(f"Video '{video_id}' is ready.")
            except Exception as e:
                print(f"Error loading video '{video_id}': {e}")
        elif cmd_lower.startswith("set video "):
            target = command[len("set video "):].strip()
            video_id = extract_video_id(target)
            if not video_id:
                print("Error: No video ID provided.")
                continue
            try:
                print(f"Indexing video '{video_id}'...")
                yt_rag.choose_video(video_id)
                print(f"Video '{video_id}' is ready.")
            except Exception as e:
                print(f"Error loading video '{video_id}': {e}")
        else:
            try:
                answer = yt_rag.run(command)
                print(answer)
            except Exception as e:
                print(f"Error generating answer: {e}")


if __name__ == "__main__":
    main()