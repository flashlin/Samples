// See https://aka.ms/new-console-template for more information
using VimSharpLib;

var editor = new VimEditor();
editor.Initialize();
editor.Render();

Console.ReadKey();