#### This project explores the use of deep learning to evaluate chess positions based on a single board state. 
#### Instead of relying on traditional search-based chess engines, we use a Transformer-based neural network 
#### to directly predict the game outcome (White win, Black win, or draw) from the board representation. 
#### The goal is to study whether a neural model can learn meaningful positional understanding without explicit game-tree search.


#### In order to run the project please follow the walkthrough below:  

#### create folders for game files:   
mkdir rawdata/train
mkdir rawdata/test
mkdir rawdata/val

#### download one or more game files from Lichess for training:  
wget https://database.lichess.org/standard/lichess_db_standard_rated_2016-01.pgn.zst -P rawdata/train/  
wget https://database.lichess.org/standard/lichess_db_standard_rated_2016-02.pgn.zst -P rawdata/train/  
wget https://database.lichess.org/standard/lichess_db_standard_rated_2016-03.pgn.zst -P rawdata/train/  
wget https://database.lichess.org/standard/lichess_db_standard_rated_2016-04.pgn.zst -P rawdata/train/  

#### download one or more game files from Lichess for validation:  
wget https://database.lichess.org/standard/lichess_db_standard_rated_2016-10.pgn.zst -P rawdata/val/  

#### download one or more game files from Lichess for testing:  
wget https://database.lichess.org/standard/lichess_db_standard_rated_2023-11.pgn.zst -P rawdata/test/  

#### create dataset folders:  
mkdir ds/train  
mkdir ds/test  
mkdir ds/val  

#### prepare datasets train/val/test:  
python datasets.py rawdata/train ds/train/train --max-games 500000 --states-per-game 5 --include-missing-eval-states --start-skip 20 --state-sample-method uniform  
python datasets.py rawdata/val ds/val/val --max-games 10000 --states-per-game 5 --include-missing-eval-states --start-skip 20 --state-sample-method uniform  
python datasets.py rawdata/test ds/test/test --max-games 50000 --states-per-game 5 --state-sample-method uniform  
  
#### run training:  
python -u run_train.py settings.py | tee -a logs/train_$(date +%F_%H-%M).log   

#### run analysis:  
#### choose the best model according to the log file and run the following command: 
python analyze_test.py --dataset ds/test/lichess_db_standard_rated_2023-11_test_0.pkl --eval-threshold 10 --models models/trnsfrm2__40.pth  models/hist__40.pth

#### analysis results will be generated into the analysis folder 


