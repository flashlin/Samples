$MODE = "production"
$IMAGE_TAG = "sqlite-front:0001-$MODE"
docker build -t $IMAGE_TAG --build-arg MODE=$MODE --progress=plain .