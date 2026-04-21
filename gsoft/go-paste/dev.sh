mkdir -p ../outputs && cd src && go build -o ../../outputs/go-paste . && cd .. && ../outputs/go-paste "$@"
