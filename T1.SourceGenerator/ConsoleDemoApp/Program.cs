using ConsoleDemoApp;
using T1.SourceGenerator;


Console.WriteLine("Hello, World!");


var a = new UserEntity();
a.Name = "Test";
a.Level = 3;
var b = new UserDto();
var c = a.ToXXX((s, t) =>
{
    t.Name = "flash";
    t.Price = 123;
});

Console.WriteLine($"{c.Name} {c.Level} {c.Price} {c.Birth}");


