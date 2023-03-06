# build the image
docker build -t predict_next_word_api .

# run a new docker container named cashman
docker run --name predict_next_word_api `
   -d -p 80:5000 `
   predict_next_word_api