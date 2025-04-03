using VimSharpLib;

namespace VimSharpLib;

public class VimCommandMode : VimInsertMode
{
    public VimCommandMode(IKeyHandler keyHandler, IVimFactory vimFactory) 
        : base(keyHandler, vimFactory)
    {
    }
    
    protected override void HandleEnterKey(List<ConsoleKeyInfo> keys)
    {
        HandleEscape(keys);
    }

    protected override void HandleEscape(List<ConsoleKeyInfo> keys)
    {
        Instance.OnClose?.Invoke();
    }

    protected override void MoveCursorUp(List<ConsoleKeyInfo> keys)
    {
        // 不執行任何動作
    }

    protected override void MoveCursorDown(List<ConsoleKeyInfo> keys)
    {
        // 不執行任何動作
    }
} 