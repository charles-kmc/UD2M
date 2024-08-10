mkdir -p $BASE_PATH/datasets/  # create datasets folder if it does not exist
cd $BASE_PATH/datasets  # move to datasets folder

# ------- CelebA-HQ --------- #
mkdir -p celeba_hq
mkdir -p download_scripts
cd download_scripts
wget https://github.com/clovaai/stargan-v2/raw/master/download.sh

bash download.sh celeba-hq-dataset
mkdir -p celeba_hq_train_split
mv data/celeba_hq/train/male/* celeba_hq_train_split/
mv data/celeba_hq/train/female/* celeba_hq_train_split/
mkdir -p celeba_hq_eval_split
mv data/celeba_hq/val/male/* celeba_hq_eval_split/
mv data/celeba_hq/val/female/* celeba_hq_eval_split/
rm -rf data/
mv celeba_hq_train_split ../celeba_hq/
mv celeba_hq_eval_split ../celeba_hq
cd ..

