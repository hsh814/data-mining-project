# Advanced data mining project

## Quick start
```bash
# Unzip data
./setup.sh
# Install dependencies
pip3 install -r requirements.txt

# Run project
python3 communityDetection.py -n ./data/TC1-1/1-1.dat
python3 evaluation.py -n ./data/TC1-1/1-1.dat
```


## Options
### communityDetection.py
- `n`: dataset
- `m`: method - you can choose between `scan`(default) and `louvain`
- `o`: output community file location
- `e`: The epsilon value for `scan` algorithm (used for finding epsilon-neighborhood) - default is 0.25
- `c`: core threshold (myu value in `scan` algorithm) - default is 3
- `u`: use modularity (use modularity to assign remaining nodes to cluster) - default is false
```bash
python3 communityDetection.py -n ./data/TC1-6/1-6.dat
# This command is equal to command below
python3 communityDetection.py -n ./data/TC1-6/1-6.dat -m scan -e 0.25 -c 3 -o ./data/TC1-6/1-6.cmty 
```

### evaluation.py
- `n`: dataset
- `r`: resulting community file from communityDetection.py
```bash
python3 evaluation.py -n ./data/TC1-6/1-6.dat
# This command is equal to command below
python3 evaluation.py -n ./data/TC1-6/1-6.dat -r ./data/TC1-6/1-6.cmty 
```
