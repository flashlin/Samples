using ConsoleDemoApp;
using T1.SourceGenerator;

Console.WriteLine("Hello, World!");


var a = new UserEntity();
a.Name = "Test";
a.Level = 3;
var b = new UserDto();

var c = a.ToXXX();
Console.WriteLine(c.Name);




var client = new SamApiClient(null);

