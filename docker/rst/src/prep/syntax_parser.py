import subprocess

from paths import STANFORD_PARSER_PATH


class SyntaxParser:

    def __init__(self):
        cmd = 'java -Xmx1000m -cp "%s/*" ParserDemo' % STANFORD_PARSER_PATH.as_posix()

        self.syntax_parser = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        init = self.syntax_parser.stderr.readline()
        if not init.startswith('Loading parser from serialized file'):
            raise OSError('Could not create a syntax parser subprocess, error info:\n%s' % init)

    def parse_sentence(self, s):
        self.syntax_parser.stdin.write("%s\n" % s.strip())
        self.syntax_parser.stdin.flush()

        # Read stderr anyway to avoid problems
        cur_line = "debut"

        finished_penn_parse = False
        penn_parse_result = ""
        dep_parse_results = []
        while cur_line != "":
            cur_line = self.syntax_parser.stdout.readline()
            # Check for errors
            if cur_line.strip() == "SENTENCE_SKIPPED_OR_UNPARSABLE":
                raise Exception("Syntactic parsing of the following sentence failed:" + s + "--")

            if cur_line.strip() == '':
                if not finished_penn_parse:
                    finished_penn_parse = True
                else:
                    break
            else:
                if finished_penn_parse:
                    dep_parse_results.append(cur_line.strip())
                else:
                    penn_parse_result = penn_parse_result + cur_line.strip()

        return penn_parse_result, '\n'.join(dep_parse_results)

    def poll(self):
        """
        Checks that the parser process is still alive
        """
        if self.syntax_parser is None:
            return True
        else:
            return self.syntax_parser.poll() is not None

    def unload(self):
        if not self.syntax_parser.poll():
            # self.syntax_parser.kill() # Only in Python 2.6+
            self.syntax_parser.stdin.close()
            self.syntax_parser.stdout.close()
            self.syntax_parser.stderr.close()
