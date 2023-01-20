# choco install antlr4 -y
# HelloLexer.cs
# HelloParser.cs
# HelloListener.cs
# HelloBaseListener.cs
# HelloVisitor.cs       (要加上 -visitor 參數才會產生這個檔)
# HelloBaseVisitor.cs   (要加上 -visitor 參數才會產生這個檔)
# antlr4.exe Tsql.g4 -listener -visitor -package T1.SqlDom.Tsql -Dlanguage=CSharp -o ./T1.SqlDom/Tsql
antlr4.exe TsqlLexer.g4 -listener -visitor -package T1.SqlDom.Tsql -Dlanguage=CSharp -o ./T1.SqlDom/Tsql
antlr4.exe TsqlParser.g4 -listener -visitor -package T1.SqlDom.Tsql -Dlanguage=CSharp -o ./T1.SqlDom/Tsql