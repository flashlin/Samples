using System.Collections.Generic;

namespace T1.Roslyn;

public class CompileError
{
	public List<string> Errors { get; set; } = new();
}