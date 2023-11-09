How to Upgrade ELK ?
0. check release note for breaking change
	- might need to change config file for outdated configuration
1. RDP 進入跳板機, 名稱開頭 Tifa 的就可以
2. open terminal （推薦 MobaXterm Personal Edition）
3. ssh 10.26.11.77，密碼是 ppuff 密碼
4. sudo docker pull elasticsearch:<version>
   版本可以看 https://hub.docker.com/_/elasticsearch/tags ，不要用版本最後一位是 0 的
5. 打包，這步驟會要一段時間
	sudo docker save elasticsearch:8.10.3 | gzip > es8.10.3.tgz
6. 因為 dockerhub 有限制特定網域可以拉 image 的次數，所以要先 下載，打包，再傳到其他需要適用的機器上
7. 用 scp 把檔案傳到其他機器，一共 9 台機器，ip 尾數是 70-78
8. scp es8.10.3.tgz 10.26.11.78:/tmp
9. 再開新的 terminal，ssh進 .78
	到 /tmp 目錄，解壓檔案放進 docker
	gunzip -c es8.10.3.tgz | sudo docker load

10. 拉 kibana:<version>，版本跟 elasticsearch 一樣
11. 到每臺機器，修改 docker file
	sudo vi /srv/elasticsearch/docker-compose.yaml
	把 image 的版本更改成 8.10.3 (新的版號）
11. 到 .78 ,執行 sudo docker-compose -f /srv/elasticsearch/docker-compose.yaml up -d --force-recreate
12. 到 http://10.26.11.78:9000 可以看到 node 下線再重新上線
13. 到 http://xros.coreop.net 設定downtime, 不然 alert 會叫
14. 
15. upgrade filebeats
	go to http://git.coreop.net/b2c/beats-gcp
16. change gitlab-ci yml version at up
17. release k8s-stg
18. release fleet-z
19. go to kibana website > fleet
	should start to see new version
	check for offline agent, and manually unenroll (immediately)