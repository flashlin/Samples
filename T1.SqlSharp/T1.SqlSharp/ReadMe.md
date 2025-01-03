
# T1.SqlSharp
## SqlParser 
Only support T-SQL Syntax:
* Create Table
* Select
```csharp
var sql = "SELECT * FROM table WHERE column = 'value'";
var sqlParser = new SqlParser(sql);
var parsedSql = sqlParser.Parse();
```