namespace VimSharpLib;

public interface IKeyHandler
{
    void InitializeKeyHandlers(Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>> keyPatterns);
    void PressKey(ConsoleKeyInfo keyInfo);
    void WaitForInput();
    void Clear();
    string GetKeyBufferString();
    void AddOnKeyPress(IKeyPattern keyPattern, Action<IProgress> action);
    bool HandleUserKeyPress(ConsoleKeyInfo keyInfo);
    void SetEditor(VimEditor editor);
}