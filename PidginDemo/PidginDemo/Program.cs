// See https://aka.ms/new-console-template for more information

using System.Text.Json;
using PidginDemo.LinqExpressions;

var linqText = "from tb1 in customer select tb1";
var expr = LinqParser.ParseOrThrow(linqText) as SelectExpr;

var options = new JsonSerializerOptions
{
    PropertyNameCaseInsensitive = true
};

Console.WriteLine($"Hello, World! @{expr.AliasTable}");