
# install setuptools and pip for package management
sudo apt-get install python-setuptools
#
## install pip
sudo apt-get install python-pip python-dev build-essential
#
## Now install other packages using pip
sudo pip install numpy  # install numpy
sudo pip install pandas # install pandas
sudo pip install qpython # q connection
#
## Install Q/KDB 
wget https://kx.com/347_d0szre-fr8917_llrsT4Yle-5839sdX/3.5/linuxx86.zip
mv linuxx86.zip ~
cp -r database ~
cd ~
unzip linuxx86.zip
sudo apt-get install libc6-i386
sudo apt-get install rlwrap


# Load database
QHOME=~/q rlwrap -r ~/q/l32/q database/database.q -p 10000 &
