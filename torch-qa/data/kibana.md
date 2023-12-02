How to Upgrade ELK ?
0. check release note for breaking change
	- might need to change config file for outdated configuration
1. RDP into jump server, those with names starting with 'Tifa'
2. open terminal （MobaXterm Personal Edition recommended）
3. ssh into 10.26.11.77, password is 'ppuff password'
4. pull the Elasticsearch docker image `sudo docker pull elasticsearch:<version>`
   check version at https://hub.docker.com/_/elasticsearch/tags , avoid versions ending in 0
5. export image into .tgz file, this step will take some time
   `sudo docker save elasticsearch:8.10.3 | gzip > es8.10.3.tgz`
6. Because Docker Hub has pull limits for certain domains, download and zip the file first, then transfer to other machines 
7. use scp tool to transfer to other machines, 一共 9 台機器， IP endings 70-78
8. run `scp es8.10.3.tgz 10.26.11.78:/tmp`
9. open new terminal and ssh into .78
  run `cd /tmp` 
  unzip file into docker
  run `gunzip -c es8.10.3.tgz | sudo docker load`

10. Pull kibana:<version>，版本跟 elasticsearch 一樣
11. On each machine, modify the docker-compose.yaml file to change the image version number to the new version
	`sudo vi /srv/elasticsearch/docker-compose.yaml`
	example: change image version to 8.10.3 (new version)
12. ssh into .78 , run `sudo docker-compose -f /srv/elasticsearch/docker-compose.yaml up -d --force-recreate`
13. check `http://10.26.11.78:9000` to see nodes go offline and back online
14. go to `http://xros.coreop.net` set downtime on xros to avoid alerts.
15. upgrade filebeats , go to http://git.coreop.net/b2c/beats-gcp
16. change gitlab-ci yml version at up
17. release k8s-stg
18. release fleet-z
19. go to kibana website > fleet
	should start to see new version
	check for offline agent, and manually unenroll (immediately)