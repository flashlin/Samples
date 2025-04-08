using VimSharpLib;

namespace VimSharpApp;

public class Main
{
    private readonly IVimFactory _vimFactory;

    public Main(IVimFactory vimFactory)
    {
        _vimFactory = vimFactory;
    }

    public void Run()
    {
        var consoleDevice = new ConsoleDevice();
        var editor1 = _vimFactory.CreateEditor<VimEditor>(consoleDevice);
        editor1.OpenText($"""
        Hello, World!
        123
        Say Hello, 閃電!
        Example
        Your name is 閃電俠
        """);
        editor1.Context.IsLineNumberVisible = true;
        editor1.Context.IsStatusBarVisible = true;
        editor1.Context.SetViewPort(1, 1, consoleDevice.WindowWidth-2, 10);

        var editor2 = _vimFactory.CreateEditor<VimEditor>(consoleDevice);
        editor2.OpenText("Example2: Editor2!");
        editor2.Context.SetViewPort(1, 12, consoleDevice.WindowWidth-2, 10);

        editor1.AddOnKeyPress(new UserKeyPattern{
            KeyChar = 'j',
            IsCtrl = false,
            IsAlt = true,
            }, (progress) =>
        {
            var converter = new TextToCsvConverter();
            progress.ShowProgress("Start Convert to CSV...");
            var text = editor1.GetText();
            progress.ShowProgress("Converting to CSV...");
            var csv = converter.Convert(text);
            progress.ShowProgress("Converting to CSV End...");
            editor2.OpenText(csv);
            progress.ShowProgress("Output to CSV Done.");
        });


        var vim = new VimSharp(consoleDevice);
        vim.AddEditor(editor1);
        vim.AddEditor(editor2);
        vim.FocusEditor(editor1);
        vim.Run();
    }
} 