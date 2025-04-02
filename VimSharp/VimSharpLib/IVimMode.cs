using System.Text;

namespace VimSharpLib;

public interface IVimMode
{
    void WaitForInput();
    VimEditor Instance { get; set; }
    void PressKey(ConsoleKeyInfo keyInfo);
    void AfterRender(StringBuilder outputBuffer);
    void Render(ColoredCharScreen screenBuffer);
} 