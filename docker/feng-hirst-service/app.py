#!/usr/bin/env python3
import os
import tempfile

import hug
import sh
from falcon import HTTP_500, HTTP_400

PARSER_PATH = "/opt/feng-hirst-rst-parser/src"
PARSER_EXECUTABLE = (
    "parser_wrapper.py"  # Feng/Hirst uses Python 2, but our API is in Python 3
)


@hug.post("/api/rst/parse")
def call_parser(body, response):
    parser = sh.Command(os.path.join(PARSER_PATH, PARSER_EXECUTABLE))

    if body and "input" in body:
        input_file_content = body["input"]
        with tempfile.NamedTemporaryFile() as input_file:
            input_file.write(input_file_content)
            input_file.flush()
            try:
                result = parser(input_file.name, _cwd=PARSER_PATH)
                with tempfile.NamedTemporaryFile(delete=False) as output_file:
                    output_file.write(result.stdout)
                    output_file.flush()
                    return result.stdout

            except sh.ErrorReturnCode_1 as err:
                response.status = HTTP_500
                trace = str(err.stderr, "utf-8")
                error_msg = "{0}\n\n{1}".format(err, trace).encode("utf-8")

                with tempfile.NamedTemporaryFile(delete=False) as error_file:
                    error_file.write(error_msg)
                    error_file.flush()
                    return error_msg

    else:
        response.status = HTTP_400
        return {"body": body}
