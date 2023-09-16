// See https://aka.ms/new-console-template for more information

using PidginDemo.LinqExpressions;

var linqText = "from tb1 in customer select tb1";
var expr = LinqParser.ParseOrThrow(linqText);

Console.WriteLine("Hello, World!");