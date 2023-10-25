from flask import Flask, request
from slack import WebClient
from slackeventsapi import SlackEventAdapter

app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(slack_signing_secret, "/slack/events", app)
slack_web_client = WebClient(token=slack_bot_token)

# app_mentions:read Enable the robot to have the ability to read mentioned messages, recognize and interpret relevant content, and respond automatically
# chat:write Enhance the bot service to enable message delivery to a specific channel.



@app.route("/", methods=['POST'])
def post_handling():
    data = request.get_json()
    logger.info(f"Incoming payload\n{data}")

    # Slack need to check application is valid
    if "challenge" in data.keys():
        logger.info("Check accessibility")
        return data["challenge"]

    # Handle slack events
    else:
        event = data.get("event")
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text")
        event_type = event.get("type")
        ts = event.get("ts") if event.get("thread_ts") is None else event.get("thread_ts")
        if event_type == "app_mention":
        # Do what you want to do here

        response = slack_web_client.chat_postMessage(channel=channel_id, text=return_message, thread_ts=ts)