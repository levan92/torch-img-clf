# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

# pretrained weights
mkdir -p pretrained
wget -nc https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth -P pretrained/
wget -nc https://download.pytorch.org/models/resnet50-19c8e357.pth -P pretrained/
wget -nc https://download.pytorch.org/models/resnet18-5c106cde.pth -P pretrained/

# trained weights
# wget -nc --no-check-certificate "https://drive.google.com/uc?export=download&id=1oVL_OOBnAIqUNDOqws-ZTFNZhtLscVZX" -O resnet50_bestval0.813_vanilla.pth
# wget -nc --no-check-certificate "https://drive.google.com/uc?export=download&id=1qIGqJA6Y5PDg2dRgaTPbPLh3BsO-dtJD" -O resnext50_bestval0.816_vanilla.pth
# gdown -O resnet18_bestval_loss0.166_acc0.947_ep10of13_bigmix-bs4-lr0.1.pth https://drive.google.com/uc?id=1-7AldjlQZULtweOaFRcZK_OqeePrtABw