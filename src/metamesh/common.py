from enum import Enum

class MessageType(Enum):
    PLAIN = "plain"
    JSON = "json"
    ERROR = "error"

def format_message(message_type, content):
    return {
        "type": message_type,
        "content": content
    }

def timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")