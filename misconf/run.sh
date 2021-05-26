#  chmod +x run.sh 使用前
if [[ $1 == 'train' ]]
then
  echo "****************Now we are going to $1 !************************"
  python ../main.py train --train-data-root=../data/train --load-model-path=None --lr=0.005 --batch-size=32 --model='ResNet34' --max-epoch=20
elif [[ $1 == 'test' ]]
then
  echo "****************Now we are going to $1 !************************"
  python main.py test --data-root=./data/test --load-model-path=None
elif [[ $1 == 'visual' ]]
then
  echo "****************Now we are going to $1 !************************"
  tensorboard --logdir ../logfile/runs --port 8889
else
  echo "You should add train | test | visual behind the .sh"
fi

