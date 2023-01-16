using ConsoleDemoApp;
using T1.SourceGenerator;

Console.WriteLine("Hello, World!");


var a = new UserEntity();
a.Name = "Test";
var b = new UserDto();

var c = a.ToUserDto();
Console.WriteLine(c.Name);

