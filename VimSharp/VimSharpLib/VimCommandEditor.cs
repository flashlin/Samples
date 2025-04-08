using VimSharpLib;

namespace VimSharpLib;

public class VimCommandEditor : VimEditor
{
    private ColoredCharScreen? _backupScreen;

    public VimCommandEditor(IVimFactory vimFactory, IConsoleDevice console, IKeyHandler keyHandler) 
        : base(vimFactory, keyHandler)
    {
        var viewWidth = (int)(console.WindowWidth * 0.8);
        var viewX = console.WindowWidth / 2 - viewWidth / 2;
        var viewY = 5;
        Context.SetViewPort(viewX, viewY, viewWidth, 1);
        Mode = vimFactory.CreateVimMode<VimCommandMode>(this);
    }

    public override void Render(ColoredCharScreen screenBuffer)
    {
        // 如果是第一次渲染，備份原始螢幕內容
        if (_backupScreen == null)
        {
            var viewPort = Context.ViewPort;
            // 創建比 ViewPort 大一圈的備份區域
            _backupScreen = new ColoredCharScreen(viewPort.Height + 2, viewPort.Width + 2);
            
            // 複製 ViewPort 範圍的螢幕內容，包含周圍一圈
            for (int y = 0; y < viewPort.Height + 2; y++)
            {
                for (int x = 0; x < viewPort.Width + 2; x++)
                {
                    _backupScreen[y, x] = screenBuffer[viewPort.Y + y - 1, viewPort.X + x - 1];
                }
            }
        }

        base.Render(screenBuffer);
    }

    public void RestoreScreen(ColoredCharScreen bufferScreen)
    {
        if (_backupScreen == null) return;

        var viewPort = Context.ViewPort;
        // 將備份的內容還原到指定的緩衝區，包含周圍一圈
        for (int y = 0; y < viewPort.Height + 2; y++)
        {
            for (int x = 0; x < viewPort.Width + 2; x++)
            {
                bufferScreen[viewPort.Y + y - 1, viewPort.X + x - 1] = _backupScreen[y, x];
            }
        }
    }
} 