module load cuda/9.0 anaconda3
source /scinet/sgc/Applications/anaconda3/bin/activate script 
while read requirement; do conda install --yes $requirement; done < requirements.txt
