sudo apt-get -y install autojump
echo '. /usr/share/autojump/autojump.sh'>>~/.bashrc

echo 'j --purge  去除不存在的路徑'
echo 'j --stat   顯示路徑的使用次數'
echo 'j -i 增加權重'
echo 'j -d 減少權重'
echo 'please restart WSL2 to take effect'