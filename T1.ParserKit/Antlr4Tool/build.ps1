# antlr4 -Dlanguage=CSharp tsql.g4
# java -jar antlr-4.11.1-complete.jar -Dlanguage=CSharp -visitor TSQL.g4
java -jar antlr-4.11.1-complete.jar -Dlanguage=CSharp -visitor -o ./dist tsql/TSqlLexer.g4
java -jar antlr-4.11.1-complete.jar -Dlanguage=CSharp -visitor -o ./dist tsql/TSqlParser.g4
# Copy-Item -Path ./*.cs -Destination ../T1.ParserKit/AntlrTsql -Force
Copy-Item -Path ./dist/*.cs -Destination ../T1.ParserKit/AntlrTsql -Force
