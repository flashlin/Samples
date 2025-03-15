namespace VimSharp
{
    public interface IVimMode
    {
        void WaitForInput();
        VimEditor Instance { get; set; }
    }
} 