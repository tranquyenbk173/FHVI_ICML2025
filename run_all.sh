#!/bin/bash

# Directory containing the config files
CONFIG_DIR="/scratch/hvp2011/t/FHVI/final_configs/"

# Directory for logs
LOG_DIR="/scratch/hvp2011/t/FHVI/logs_sbatch/"

for config_file in "$CONFIG_DIR"/*.yaml; do
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=$LOG_DIR/%A.out
#SBATCH --error=$LOG_DIR/%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tuantruong.shecodes@gmail.com


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

source activate /vast/hvp2011/t_envs/peft
cd /scratch/hvp2011/t/FHVI

echo $config_file

python main.py fit --config  "$config_file"
EOT
done
