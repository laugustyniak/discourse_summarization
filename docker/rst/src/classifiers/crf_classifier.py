import subprocess
from os.path import join, exists

import paths


class CRFClassifier:
    def __init__(self, name, model_type, model_path, model_file, verbose):
        self.verbose = 1
        self.name = name
        self.type = model_type
        self.model_fname = model_file
        self.model_path = model_path

        if not exists(join(self.model_path, self.model_fname)):
            print('The model path %s for CRF classifier %s does not exist.' % (
                join(self.model_path, self.model_fname), name))
            raise OSError('Could not create classifier subprocess')

        self.getConsole()

    def getConsole(self):
        self.classifier_cmd = '%s/crfsuite-stdin tag -pi -m %s -' % (
            paths.CRFSUITE_PATH, join(self.model_path, self.model_fname)
        )
        print(self.classifier_cmd)
        self.classifier = subprocess.Popen(
            self.classifier_cmd, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )

        if self.classifier.poll():
            raise OSError(
                'Could not create classifier subprocess, with error info:\n%s' % self.classifier.stderr.readline())

        return self.classifier

    def classify(self, vectors):
        out, err = self.getConsole().communicate('\n'.join(vectors) + "\n\n")
        print('out', out)
        print('err', err)
        lines = out.split("\n")
        print('lines', lines)

        if self.classifier.poll():
            raise OSError('crf_classifier subprocess died')

        predictions = []
        for line in lines[1:]:
            line = line.strip()
            if line != '':
                fields = line.split(':')
                label = fields[0]
                prob = float(fields[1])
                predictions.append((label, prob))

        seq_prob = float(lines[0].split('\t')[1])
        return seq_prob, predictions

    def poll(self):
        """
        Checks that the classifier processes are still alive
        """
        if self.classifier is None:
            return True
        else:
            return self.classifier.poll() != None

    def unload(self):
        if self.classifier and not self.poll():
            self.classifier.stdin.write('\n')
