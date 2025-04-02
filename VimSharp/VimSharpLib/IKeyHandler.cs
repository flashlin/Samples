namespace VimSharpLib;

public interface IKeyHandler
{
    void InitializeKeyHandlers(Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>> keyPatterns);
    void PressKey(ConsoleKeyInfo keyInfo);
    void WaitForInput();
}