Run the below code to test input/output. the test sentence is in inputdata.py
```
module load anaconda3/2020.11
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate spanbert
cd "/home/ruicheng/spanbert-coref-master_final/"
export data_dir=./data
./setup_all.sh

python -W ignore inputdata.py
python -W ignore predict.py spanbert_base sample.in.json out.txt
```

Run the below code to show the F1 score:
```
