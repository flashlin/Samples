// See https://aka.ms/new-console-template for more information
using VimSharpLib;

var editor1 = new VimEditor();
editor1.OpenText($"""
Hello, World!
123
Say Hello, 閃電!
Example
Your name is 閃電俠
""");
editor1.Context.IsLineNumberVisible = true;
editor1.Context.IsStatusBarVisible = true;
editor1.Context.SetViewPort(1, 1, 40, 5);

var editor2 = new VimEditor();
editor2.OpenText("Example2: Editor2!");
editor2.Context.SetViewPort(20, 12, 40, 10);

var vim = new VimSharp();
vim.AddEditor(editor1);
vim.AddEditor(editor2);
vim.FocusEditor(editor1);
vim.Run();

