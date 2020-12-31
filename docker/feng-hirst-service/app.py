import os
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
import sh

parser_app = FastAPI()

PARSER_PATH = "/opt/feng-hirst-rst-parser/src"
PARSER_EXECUTABLE = (
    "parser_wrapper.py"  # Feng/Hirst uses Python 2, but our API is in Python 3
)


class Request(BaseModel):
    text: str


@parser_app.post("/api/rst/parse")
async def call_parser(request: Request):
    parser = sh.Command(os.path.join(PARSER_PATH, PARSER_EXECUTABLE))
    print(os.path.join(PARSER_PATH, PARSER_EXECUTABLE))

    with tempfile.NamedTemporaryFile("w+t") as input_file:
        input_file.write(request.text)
        input_file.flush()
        try:
            result = parser(input_file.name, _cwd=PARSER_PATH)
            return result.stdout

        except Exception:
            return ''
