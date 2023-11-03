from datetime import datetime

from pydantic import BaseModel

from obj_utils import dict_to_dynamic_object, clone_to_dynamic_object


class Req(BaseModel):
    message: str


a = Req(message="123")

b = clone_to_dynamic_object(a, {
    "conversation_type": "message"
})


print(f"{b.__dict__=}")
print(f"{b.conversation_type=}")
