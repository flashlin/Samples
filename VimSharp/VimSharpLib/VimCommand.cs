using VimSharpLib;

namespace VimSharpLib;

public class VimCommand : VimEditor
{
    private ColoredCharScreen? _backupScreen;

    public VimCommand(IConsoleDevice console) : base(console)
    {
        var viewWidth = (int)(console.WindowWidth * 0.8);
        var viewX = console.WindowWidth / 2 - viewWidth / 2;
        var viewY = 5;
        Context.SetViewPort(viewX, viewY, viewWidth, 1);
        Mode = new VimCommandMode(this);
    }

    public override void Render(ColoredCharScreen? screenBuffer = null)
    {
        // 如果是第一次渲染，備份原始螢幕內容
        if (_backupScreen == null)
        {
            var viewPort = Context.ViewPort;
            _backupScreen = new ColoredCharScreen(viewPort.Height, viewPort.Width);
            
            // 複製 ViewPort 範圍的螢幕內容
            var currentBuffer = screenBuffer ?? CreateScreenBuffer();
            for (int y = 0; y < viewPort.Height; y++)
            {
                for (int x = 0; x < viewPort.Width; x++)
                {
                    _backupScreen[y, x] = currentBuffer[viewPort.Y + y, viewPort.X + x];
                }
            }
        }

        base.Render(screenBuffer);
    }

    public void RestoreScreen(ColoredCharScreen bufferScreen)
    {
        if (_backupScreen == null) return;

        var viewPort = Context.ViewPort;
        // 將備份的內容還原到指定的緩衝區
        for (int y = 0; y < viewPort.Height; y++)
        {
            for (int x = 0; x < viewPort.Width; x++)
            {
                bufferScreen[viewPort.Y + y, viewPort.X + x] = _backupScreen[y, x];
            }
        }
    }
} 