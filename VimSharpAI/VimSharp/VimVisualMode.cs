namespace VimSharp
{
    public class VimVisualMode : IVimMode
    {
        public VimEditor Instance { get; set; }
        private int _selectionStartX;
        private int _selectionStartY;

        public VimVisualMode()
        {
            // 初始化選擇起點為當前游標位置
            _selectionStartX = -1;
            _selectionStartY = -1;
        }

        public void WaitForInput()
        {
            // 如果尚未設置選擇起點，則設置為當前位置
            if (_selectionStartX < 0 || _selectionStartY < 0)
            {
                _selectionStartX = Instance.GetActualTextX();
                _selectionStartY = Instance.GetActualTextY();
            }

            var keyInfo = Instance.Console.ReadKey(intercept: true);
            
            // 根據按鍵執行相應操作
            switch (keyInfo.Key)
            {
                case ConsoleKey.Escape:
                    // ESC 返回普通模式
                    Instance.StatusBar = new ConsoleText("-- NORMAL --");
                    Instance.Mode = new VimNormalMode { Instance = Instance };
                    break;
                case ConsoleKey.H:
                    Instance.MoveCursorLeft();
                    break;
                case ConsoleKey.J:
                    Instance.MoveCursorDown();
                    break;
                case ConsoleKey.K:
                    Instance.MoveCursorUp();
                    break;
                case ConsoleKey.L:
                    Instance.MoveCursorRight();
                    break;
                case ConsoleKey.Y:
                    // 複製選中內容
                    YankSelection();
                    Instance.StatusBar = new ConsoleText("-- NORMAL --");
                    Instance.Mode = new VimNormalMode { Instance = Instance };
                    break;
            }
        }

        private void YankSelection()
        {
            // 實際應用中這裡可以實現內容複製功能
            // 由於需求規範中沒有詳細說明，這裡只是一個示例
            int startY = Math.Min(_selectionStartY, Instance.GetActualTextY());
            int endY = Math.Max(_selectionStartY, Instance.GetActualTextY());
            
            // 更新狀態欄，顯示已選擇的行數
            Instance.StatusBar = new ConsoleText($"Selected {endY - startY + 1} lines");
        }
    }
} 