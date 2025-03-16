namespace VimSharpLib.Tests
{
    /// <summary>
    /// 模擬的控制台設備，用於測試
    /// </summary>
    public class MockConsoleDevice : IConsoleDevice
    {
        /// <summary>
        /// 獲取控制台視窗寬度
        /// </summary>
        public int WindowWidth { get; set; } = 80;
        
        /// <summary>
        /// 獲取控制台視窗高度
        /// </summary>
        public int WindowHeight { get; set; } = 25;
        
        /// <summary>
        /// 設置光標位置
        /// </summary>
        /// <param name="left">左邊距</param>
        /// <param name="top">上邊距</param>
        public void SetCursorPosition(int left, int top)
        {
            // 在測試中不需要實際設置光標位置
        }
        
        /// <summary>
        /// 寫入文本到控制台
        /// </summary>
        /// <param name="value">要寫入的文本</param>
        public void Write(string value)
        {
            // 在測試中不需要實際寫入文本
        }
        
        /// <summary>
        /// 讀取按鍵
        /// </summary>
        /// <param name="intercept">是否攔截按鍵</param>
        /// <returns>按鍵信息</returns>
        public ConsoleKeyInfo ReadKey(bool intercept)
        {
            // 在測試中返回一個默認的按鍵信息
            return new ConsoleKeyInfo('\0', ConsoleKey.Escape, false, false, false);
        }
    }
} 