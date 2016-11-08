import subprocess
import paths
import fcntl
import os.path

class CRFClassifier:
    def __init__(self, name, model_type, model_path, model_file, verbose):
        self.verbose = verbose
        self.name = name
        self.type = model_type
        self.model_fname = model_file
        self.model_path = model_path
        
        if not os.path.exists(os.path.join(self.model_path, self.model_fname)):
            print 'The model path %s for CRF classifier %s does not exist.' % (os.path.join(self.model_path, self.model_fname), name)
            raise OSError('Could not create classifier subprocess')
        
        self.getConsole()
        
        #self.cnt = 0
          
    def getConsole(self):
        self.classifier_cmd = '%s/crfsuite-stdin tag -pi -m %s -' % (paths.CRFSUITE_PATH, 
							 os.path.join(self.model_path, self.model_fname))
        #print self.classifier_cmd
        self.classifier = subprocess.Popen(self.classifier_cmd, shell = True, stdin = subprocess.PIPE, stderr = subprocess.PIPE, stdout = subprocess.PIPE)
        
        if self.classifier.poll():
            raise OSError('Could not create classifier subprocess, with error info:\n%s' % self.classifier.stderr.readline())
        
        return self.classifier
		  
    def nonBlockRead(self, output):
        fd = output.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        try:
            return output.read()
        except:
            return ''
			
    def classify(self, vectors):
#        print '\n'.join(vectors) + "\n\n"
        
        out, err = self.getConsole().communicate('\n'.join(vectors) + "\n\n")
        lines = out.split("\n")
		
        if self.classifier.poll():
            raise OSError('crf_classifier subprocess died')
        
        predictions = []
        for line in lines[1 : ]:
            line = line.strip()
            if line != '':
                fields = line.split(':')
                label = fields[0]
                prob = float(fields[1])
                predictions.append((label, prob))
        
        seq_prob = float(lines[0].split('\t')[1])
        
        #print seq_prob
        #print predictions		
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
            
