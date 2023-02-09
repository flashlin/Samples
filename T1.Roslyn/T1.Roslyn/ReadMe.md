Some technical documentation is at this link: http://mr-brain.github.io/

This kit provides dynamic compilation of C# code

```
public IActionResult RunCode(RunCodeViewModel vm)
{
	var code = $@"
using System;
namespace MyDynamicNs {{
	public class MyDynamicType {{
		public string Execute() {{ 
			{vm.Code}
		}}
	}}
}}";

	var roslyn = new RoslynScripting();
	var compileResult = roslyn.Compile(code);
	compileResult.Match(assembly =>
	{
		dynamic instance = assembly.CreateInstance("MyDynamicNs.MyDynamicType")!;
		vm.Result = $"{instance.Execute()}";
		return vm;
	}, compileError =>
	{
		vm.Result = string.Join("\r\n", compileError.Errors);
		return vm;
	});
	return View("Index", vm);
}
```
