using VimSharpLib;

namespace VimSharpLib;

public class VimCommand : VimEditor
{
    public VimCommand(IConsoleDevice console) : base(console)
    {
        var viewWidth = (int)(console.WindowWidth * 0.8);
        var viewX = console.WindowWidth / 2 - viewWidth / 2;
        var viewY = 5;
        Context.SetViewPort(viewX, viewY, viewWidth, 1);
        Mode = new VimInsertMode(this);
    }
} 