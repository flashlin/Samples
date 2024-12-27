
# T1.SqlSharp
## SqlParser 

```csharp
var sql = "SELECT * FROM table WHERE column = 'value'";
var sqlParser = new SqlParser(sql);
var parsedSql = sqlParser.Parse();
```