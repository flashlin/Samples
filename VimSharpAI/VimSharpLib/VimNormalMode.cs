using System.Collections.Generic;

namespace VimSharpLib
{
    public class VimNormalMode : IVimMode
    {
        public VimEditor Instance { get; set; }
        private List<ConsoleKey> _keyBuffer = new List<ConsoleKey>();
        private Dictionary<IKeyPattern, Action> _keyPatterns = new Dictionary<IKeyPattern, Action>();

        public VimNormalMode()
        {
            InitializeKeyPatterns();
        }

        private void InitializeKeyPatterns()
        {
            // 添加各種命令模式下的按鍵模式
            _keyPatterns[new SingleKeyPattern(ConsoleKey.H)] = () => Instance.MoveCursorLeft();
            _keyPatterns[new SingleKeyPattern(ConsoleKey.J)] = () => Instance.MoveCursorDown();
            _keyPatterns[new SingleKeyPattern(ConsoleKey.K)] = () => Instance.MoveCursorUp();
            _keyPatterns[new SingleKeyPattern(ConsoleKey.L)] = () => Instance.MoveCursorRight();
            _keyPatterns[new SingleKeyPattern(ConsoleKey.I)] = () => EnterInsertMode();
            _keyPatterns[new SingleKeyPattern(ConsoleKey.V)] = () => EnterVisualMode();
            _keyPatterns[new SingleKeyPattern(ConsoleKey.D4)] = () => Instance.MoveCursorToEndOfLine(); // $ 符號對應 Shift+4 (D4)
            _keyPatterns[new SingleKeyPattern(ConsoleKey.D6)] = () => Instance.MoveCursorToStartOfLine(); // ^ 符號對應 Shift+6 (D6)
        }

        private void EnterInsertMode()
        {
            Instance.StatusBar = new ConsoleText("-- INSERT --");
            Instance.Mode = new VimInsertMode { Instance = Instance };
        }

        private void EnterVisualMode()
        {
            Instance.StatusBar = new ConsoleText("-- VISUAL --");
            Instance.Mode = new VimVisualMode { Instance = Instance };
        }

        public void WaitForInput()
        {
            var keyInfo = Instance.Console.ReadKey(intercept: true);
            _keyBuffer.Add(keyInfo.Key);
            
            int matchCount = 0;
            IKeyPattern? matchedPattern = null;
            foreach (var pattern in _keyPatterns.Keys)
            {
                if (pattern.IsMatch(_keyBuffer))
                {
                    matchCount++;
                    matchedPattern = pattern;
                }
            }
            
            if (matchCount == 1 && matchedPattern != null)
            {
                _keyPatterns[matchedPattern].Invoke();
                _keyBuffer.Clear();
            }
            else if (matchCount == 0 && _keyBuffer.Count >= 3)
            {
                _keyBuffer.Clear();
            }
        }
    }

    // 簡單的單鍵模式匹配實現
    public class SingleKeyPattern : IKeyPattern
    {
        private ConsoleKey _key;

        public SingleKeyPattern(ConsoleKey key)
        {
            _key = key;
        }

        public bool IsMatch(List<ConsoleKey> keyBuffer)
        {
            return keyBuffer.Count == 1 && keyBuffer[0] == _key;
        }
    }
} 
