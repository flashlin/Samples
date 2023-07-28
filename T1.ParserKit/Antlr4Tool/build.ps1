java -jar antlr-4.11.1-complete.jar -Dlanguage=CSharp -visitor TSQL.g4
# antlr4 -Dlanguage=CSharp tsql.g4
Copy-Item -Path ./*.cs -Destination ../T1.ParserKit/AntlrTsql -Force
