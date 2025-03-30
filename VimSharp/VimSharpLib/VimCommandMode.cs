using VimSharpLib;

namespace VimSharpLib;

public class VimCommandMode : VimInsertMode
{
    public VimCommandMode(VimEditor instance) 
        : base(instance)
    {
    }
    
    protected override void HandleEnterKey(List<ConsoleKeyInfo> keys)
    {
        HandleEscape(keys);
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