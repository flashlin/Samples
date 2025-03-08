namespace VimSharpLib;

public interface IVimMode
{
    void WaitForInput();
    VimEditor Instance { get; set; }
} 