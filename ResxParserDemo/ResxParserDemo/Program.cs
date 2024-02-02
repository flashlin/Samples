// See https://aka.ms/new-console-template for more information

using ResxParserDemo;

Console.WriteLine("Hello, World!");

var parser = new ResxParser();
parser.ReadFile("D:/VDisk/GitHub/qa-pair/pre-process-data/ResxFiles/Nike/nike.zh-cn.resx");
