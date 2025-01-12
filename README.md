# ZeroLog: Zero-Label Generalizable Cross-System Log Anomaly Detection

## Project Structure
```
├─conf            # Configuration for Drain
├─datasets        # HDFS, BGL and OpenStack
├─entities        # Instances for log data and DL model.
├─logs            # Logs of method.
├─methods         # ZeroLog main entrance.      
├─module          # Anomaly detection modules, including classifier, Attention, etc.
├─outputs         # Outputs of method.
├─parsers         # Drain parser.
├─preprocessing   # Preprocessing code, data loaders and cutters.
├─representations # Log template and sequence representation.
└─utils           # Vocab for DL model and some other common utils.
```

## Datasets

We used `3` open-source log datasets, HDFS, BGL and OpenStack. 

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|-----------------|------------------------------------|------------|------------|-----------|-----------------------------------------------------------|
| HDFS            | Hadoop distributed file system log | 38.7 hours | 11,175,629 | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)                |
| BGL             | Blue Gene/L supercomputer log      | 214.7 days | 4,747,963  | 708.76MB  | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |
| OpenStack       | OpenStack infrastructure log       | N.A.       | 207,820    | 58.61MB   | [LogHub](https://github.com/logpai/loghub)                |


## Environment

Please refer to the `requirements.txt` file for package installation.

## Preparation

- **Step 1:** To run `ZeroLog` on different log data, create a directory under `datasets` folder HDFS, BGL and OpenStack.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), and put it under `datasets` folder.

## Run
- Run `methods/Source_Target.py` (make sure it has proper parameters) for zero-label generalization from Source to Target.