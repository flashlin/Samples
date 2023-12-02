Question: Aliases not available when using sudo
Answer:

Add the following line to your ~/.bashrc:
```bashrc
alias sudo='sudo '
```

Question: How to install git on ubuntu?
Answer:
```
sudo apt-get update
sudo apt-get install git
```

Question: How to use tmux tool?
Answer:
```
sudo apt-get update
sudo apt-get install tmux
```

水平分割
```bash
tmux
``` 

切換到切割頁面, 按下 `ctrl + b` 放開後接著按下 `shift + "`


垂直分割
$ ctrl + b 放開後按下 上 or 下 選擇分頁割面

$ ctrl + b 放開後接著按下 shift + %

切換到切割頁面
$ ctrl + b 放開後按下 左 or 右 選擇分頁割面

調整視窗大小
$ ctrl + b 不要放開,此時按下上下左右

離開
$ exit