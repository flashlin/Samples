using System.Text;

namespace VimSharpLib;

public interface IVimMode
{
    void WaitForInput();
    VimEditor Instance { get; }
    void PressKey(ConsoleKeyInfo keyInfo);
    void AfterRender(StringBuilder outputBuffer);
} 