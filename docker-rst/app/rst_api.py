import json
import logging
import sys

import falcon

sys.path.append('/gCRF_dist/src/')
from parse import DiscourseParser

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

try:
    parser = DiscourseParser(output_dir='/tmp')
except Exception as e:
    print('Error: {}'.format(str(e)))
    raise Exception


class RSTAPI(object):
    def max_body(limit):
        def hook(req, resp, resource, params):
            length = req.content_length
            if length is not None and length > limit:
                msg = ('The size of the request is too large. The body '
                       'must not exceed ' + str(limit) + ' bytes in length.')
                raise falcon.HTTPRequestEntityTooLarge(
                    'Request body is too large', msg)

        return hook

    @falcon.before(max_body(64 * 1024))
    def on_post(self, req, resp):
        try:
            text = req.context['doc']['text']
            resp.body = json.dumps(parser.parse(filename='tmp', parse_text=text))
        except Exception as ex:
            raise falcon.HTTPServiceUnavailable('Error', 'Service error, try ' 'again later', 30)

        resp.status = falcon.HTTP_200  # This is the default status
        resp.set_header('X-Powered-By', 'sentimentAPI')

    def on_get(self, req, resp):
        try:
            resp.body = "Testujemy!"
        except Exception as ex:
            logger.error(repr(ex))
            raise falcon.HTTPServiceUnavailable('Error', 'Service error, ' 'try again later: {}'.format(repr(ex)), 30)

        resp.status = falcon.HTTP_200  # This is the default status
        resp.set_header('X-Powered-By', 'rst_parsing')


class ErrorHandler(Exception):
    @staticmethod
    def handle(ex, req, resp, params):
        raise falcon.HTTPError(falcon.HTTP_725, 'Error', repr(ex))


class RequireJSON(object):
    def process_request(self, req, resp):
        if not req.client_accepts_json:
            raise falcon.HTTPNotAcceptable(
                'This API only supports responses encoded as JSON.')


class JSONTranslator(object):
    """
    early processor makes sure the data we get from client is valid
    and json with all fields before sent to post accepter
    """

    def process_request(self, req, resp):
        if req.content_length in (None, 0):
            # Nothing to do
            return

        body = req.stream.read()
        if not body:
            raise falcon.HTTPBadRequest('Empty request body', 'A valid JSON document is required.')

        try:
            req.context['doc'] = json.loads(body.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise falcon.HTTPError(
                falcon.HTTP_753, 'Malformed JSON',
                'Could not decode the request body. The JSON was incorrect or not encoded as UTF-8.')

        # did they pass text var
        if 'text' not in req.context['doc']:
            raise falcon.HTTPBadRequest(
                'Error',
                'Missing json var text, come on you had one var to pass')

        # did they pass an empty text var
        if not req.context['doc']['text']:
            raise falcon.HTTPBadRequest(
                'Error',
                'Missing empty var text, come on you had one var to pass')

    def process_response(self, req, resp, resource):
        if 'result' not in req.context:
            return

        resp.body = json.dumps(req.context['result'])


# falcon.API instance
app = falcon.API(middleware=[
    RequireJSON(),
    JSONTranslator(),
])

app.add_error_handler(ErrorHandler, ErrorHandler.handle)

# long-lived class instances
rstAPI = RSTAPI()

app.add_route('/api/rst/parse', rstAPI)
