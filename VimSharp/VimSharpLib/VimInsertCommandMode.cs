using VimSharpLib;

namespace VimSharpLib;

public class VimInsertCommandMode : VimInsertMode
{
    public VimInsertCommandMode(VimEditor instance) 
        : base(instance)
    {
    }
    
    protected override void HandleEnterKey(List<ConsoleKeyInfo> keys)
    {
        HandleEscape(keys);
    }
} 