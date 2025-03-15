// See https://aka.ms/new-console-template for more information
using VimSharpLib;

// Console.SetCursorPosition(10,2);
// // 使用控制碼顯示游標
// Console.Write("\x1b[?25h");
// Console.ReadKey();
// Environment.Exit(0);

var editor1 = new VimEditor();
editor1.SetText($"""
Hello, World!
123
Say Hello
Example 3, Title.
Your name is ?
""");
editor1.Context.IsLineNumberVisible = true;
editor1.Context.IsStatusBarVisible = true;
editor1.Context.SetViewPort(1, 1, 40, 5);

var editor2 = new VimEditor();
editor2.Context.SetText(0, 0, "Example2: Editor2!");
editor2.Context.SetViewPort(20, 12, 40, 10);

var vim = new VimSharp();
vim.AddEditor(editor1);
vim.AddEditor(editor2);
vim.FocusEditor(editor1);
vim.Run();

