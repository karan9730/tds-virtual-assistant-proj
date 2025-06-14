# /// script
# requires-python = "==3.12"
# dependencies = [
#     "requests<3",
#     "rich",
# ]
# ///

import requests
import json
from datetime import datetime, timezone

with open("cookies.txt", "r") as file:
    cookie = file.read().strip()

headers = {
    "cookie": cookie,
    "User-Agent": "Mozilla/5.0"
}

output_file = "discourse_topic_list.json"
start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 4, 14, 23, 59, 59, tzinfo=timezone.utc)

page = 1

with open(output_file, "w") as out_file:
    while True:
        url = f"https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json?page={page}"
        print(f"Fetching page {page}...")

        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            print(f"Failed to fetch page {page}, status code {res.status_code}")
            break

        data = res.json()
        topics = data.get("topic_list", {}).get("topics", [])

        if not topics:
            break

        for topic in topics:
            created = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))

            if start_date <= created <= end_date:
                result = {
                    "id": topic["id"],
                    "slug": topic["slug"]
                }
                out_file.write(json.dumps(result) + "\n")

        # Stop if last topic is older than start_date
        last_topic_created = datetime.fromisoformat(
            topics[-1]["created_at"].replace("Z", "+00:00")
        )
        if last_topic_created < start_date:
            print("Hit oldest required topic. Stopping.")
            break

        page += 1


# Step 2: Fetch each topic's full content
topic_list_path = "discourse_topic_list.json"
chunks_out_path = "chunks.json"

print("Fetching full topics...")

with open(topic_list_path, "r") as topic_file, open(chunks_out_path, "w") as chunks_file:
    for line in topic_file:
        try:
            topic = json.loads(line)
            topic_id = topic["id"]
            slug = topic["slug"]

            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}.json"
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                print(f"Failed to fetch topic {topic_id}")
                continue

            data = res.json()
            posts = data.get("post_stream", {}).get("posts", [])
            for post in posts:
                post_number = post["post_number"]
                cooked = post.get("cooked", "")
                if not cooked.strip():
                    continue  # skip empty posts

                # Determine URL format
                if post_number == 1:
                    post_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}"
                else:
                    post_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{post_number}"

                # Save in expected line-by-line JSON object format
                entry = {
                    "id": f"{topic_id}#{post_number}",
                    "url": post_url,
                    "content": cooked
                }
                chunks_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"Error processing topic: {e}")