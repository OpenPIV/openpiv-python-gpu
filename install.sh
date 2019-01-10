module load cuda/9.0 anaconda3
conda create --yes --name multiprocess python=2.7
source /scinet/sgc/Applications/anaconda3/bin/activate multiprocess 
while read requirement; do conda install --yes $requirement; done < requirements.txt
