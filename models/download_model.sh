FILE=$1

MODEL_FILE=./models/${FILE}_MDMM.pth
URL=http://vllab.ucmerced.edu/hylee/DRIT/models/${FILE}_MDMM.pth

wget -N $URL -O $MODEL_FILE
