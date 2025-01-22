## License

This project is licensed under the terms of the [GNU General Public License v3.0](LICENSE).

You can redistribute it and/or modify it under the terms of the GPL-3.0 license. For more details, see the [official website](https://www.gnu.org/licenses/gpl-3.0.html).

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