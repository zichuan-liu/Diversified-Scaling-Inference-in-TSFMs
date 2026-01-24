curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt install git-lfs

tar -zxvf data.tar.gz

pip install matplotlib statsmodels
pip install chronos-forecasting==1.5.3
pip install scipy scikit-learn dtaidistance
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 

git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .[torch]
cd ..
# rm -rf timesfm

# git clone https://github.com/SalesforceAIResearch/uni2ts.git
# cd uni2ts
# pip install -e '.[notebook]'
# cd ..
# # rm -rf uni2ts

