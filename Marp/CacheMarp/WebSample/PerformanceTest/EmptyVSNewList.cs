using BenchmarkDotNet.Attributes;
using WebSample.Services;

[MemoryDiagnoser]
public class EmptyVSNewList
{
	[Benchmark]
	public void Empty()
	{
		var p = new MyService();
		p.Check("flash^");
	}

	[Benchmark]
	public void NewList()
	{
		var p = new MyService();
		p.Check2("flash^");
	}
}