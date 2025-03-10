﻿// See https://aka.ms/new-console-template for more information
using VimSharpLib;

var editor1 = new VimEditor();
editor1.Context.SetText(0, 0, "Hello, World!");
editor1.Context.SetText(0, 1, "123");
editor1.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
editor1.IsStatusBarVisible = true;

var editor2 = new VimEditor();
editor2.Context.SetText(0, 0, "Example2: Editor2!");
editor2.Context.ViewPort = new ConsoleRectangle(20, 12, 40, 10);

var vim = new VimSharp();
vim.AddEditor(editor1);
vim.AddEditor(editor2);
vim.FocusEditor(editor1);
vim.Run();

