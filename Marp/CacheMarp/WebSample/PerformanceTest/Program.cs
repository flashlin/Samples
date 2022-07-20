using BenchmarkDotNet.Running;

Console.WriteLine("Hello, World!");


var summary = BenchmarkRunner.Run<EmptyVSNewList>();
Console.ReadKey();



