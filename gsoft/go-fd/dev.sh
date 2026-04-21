mkdir -p ../outputs && cd src && go build -o ../../outputs/fd . && cd .. && ../outputs/fd "$@"
