# Aspect-based Sentiment Analysis using Rhetorical Structure Analysis (RST) 

`usage: run.py [-h] [-input INPUT_FILE_PATH] [-output OUTPUT_FILE_PATH]
              [-sent_model SENT_MODEL_PATH] [-batch BATCH_SIZE]
              [-p MAX_PROCESSES]`

## Process documents

`optional arguments:
  -h, --help            show this help message and exit
  -input INPUT_FILE_PATH
                        Path to the file with documents (json, csv, pickle)
  -output OUTPUT_FILE_PATH
                        Number of processes
  -sent_model SENT_MODEL_PATH
                        path to sentiment model
  -batch BATCH_SIZE     batch size for each process
  -p MAX_PROCESSES      Number of processes
`

## Exemplary execution
`python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23`
