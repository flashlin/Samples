docker stop predict-next-word-api
docker rm predict-next-word-api
docker rmi predict-next-word-api 

Write-host "build the image"
docker build -t predict-next-word-api .

# run a new docker container named cashman
docker run --name predict-next-word-api `
   -p 8001:5000 `
   predict-next-word-api