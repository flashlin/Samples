docker stop predict_next_word_api
docker rm predict_next_word_api
docker rmi predict_next_word_api 

Write-host "build the image"
docker build -t predict_next_word_api .

# run a new docker container named cashman
docker run --name predict_next_word_api `
   -p 8001:5000 `
   predict_next_word_api