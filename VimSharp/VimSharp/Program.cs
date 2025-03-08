// See https://aka.ms/new-console-template for more information
using VimSharpLib;

var editor = new VimEditor();
editor.Context.SetText(0, 0, "Hello, World!");
editor.Context.ViewPort = new ConsoleRectangle(10, 1, Console.WindowWidth - 20, Console.WindowHeight - 2);
editor.Run();

