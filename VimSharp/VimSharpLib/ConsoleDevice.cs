namespace VimSharpLib;
using System;
using System.Text;
using System.Runtime.InteropServices;

/// <summary>
/// 控制台設備實現，封裝 System.Console
/// </summary>
public class ConsoleDevice : IConsoleDevice
{
    /// <summary>
    /// 獲取控制台視窗寬度
    /// </summary>
    public int WindowWidth => Console.WindowWidth;
    
    /// <summary>
    /// 獲取控制台視窗高度
    /// </summary>
    public int WindowHeight => Console.WindowHeight;
    
    /// <summary>
    /// 設置光標位置
    /// </summary>
    /// <param name="left">左邊距</param>
    /// <param name="top">上邊距</param>
    public void SetCursorPosition(int left, int top)
    {
        Console.Write($"\x1b[{top+1};{left+1}H");
    }

    public void SetBlockCursor()
    {
        Console.Write("\x1b[2 q");
    }
    
    public void SetLineCursor()
    {
        Console.Write("\x1b[6 q");
    }
    
    /// <summary>
    /// 寫入文本到控制台
    /// </summary>
    /// <param name="value">要寫入的文本</param>
    public void Write(string value)
    {
        Console.Write(value);
    }
    
    /// <summary>
    /// 讀取按鍵
    /// </summary>
    /// <param name="intercept">是否攔截按鍵</param>
    /// <returns>按鍵信息</returns>
    public ConsoleKeyInfo ReadKey(bool intercept)
    {
        // ESC 字符的 ASCII 碼
        const char ESC = (char)0x1B;
        
        // 緩衝區用於存儲讀取的字符
        StringBuilder inputBuffer = new StringBuilder();
        
        // 讀取第一個字符
        ConsoleKeyInfo keyInfo = Console.ReadKey(intercept);
        
        // 檢查是否是ESC字符（可能是控制序列的開始）
        if (keyInfo.KeyChar == ESC && Console.KeyAvailable)
        {
            // 記錄ESC字符
            inputBuffer.Append(keyInfo.KeyChar);
            
            // 等待一小段時間，確保所有字節都已到達
            System.Threading.Thread.Sleep(1);
            
            // 讀取控制序列的其餘部分
            while (Console.KeyAvailable)
            {
                var nextKeyInfo = Console.ReadKey(true);
                inputBuffer.Append(nextKeyInfo.KeyChar);
            }
            
            // 解析控制序列
            string sequence = inputBuffer.ToString();
            
            // 處理方向鍵
            if (sequence == $"{ESC}[A") return new ConsoleKeyInfo('\0', ConsoleKey.UpArrow, false, false, false);
            if (sequence == $"{ESC}[B") return new ConsoleKeyInfo('\0', ConsoleKey.DownArrow, false, false, false);
            if (sequence == $"{ESC}[C") return new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false);
            if (sequence == $"{ESC}[D") return new ConsoleKeyInfo('\0', ConsoleKey.LeftArrow, false, false, false);
            
            // 處理功能鍵
            if (sequence == $"{ESC}OP") return new ConsoleKeyInfo('\0', ConsoleKey.F1, false, false, false);
            if (sequence == $"{ESC}OQ") return new ConsoleKeyInfo('\0', ConsoleKey.F2, false, false, false);
            if (sequence == $"{ESC}OR") return new ConsoleKeyInfo('\0', ConsoleKey.F3, false, false, false);
            if (sequence == $"{ESC}OS") return new ConsoleKeyInfo('\0', ConsoleKey.F4, false, false, false);
            if (sequence == $"{ESC}[15~") return new ConsoleKeyInfo('\0', ConsoleKey.F5, false, false, false);
            if (sequence == $"{ESC}[17~") return new ConsoleKeyInfo('\0', ConsoleKey.F6, false, false, false);
            if (sequence == $"{ESC}[18~") return new ConsoleKeyInfo('\0', ConsoleKey.F7, false, false, false);
            if (sequence == $"{ESC}[19~") return new ConsoleKeyInfo('\0', ConsoleKey.F8, false, false, false);
            if (sequence == $"{ESC}[20~") return new ConsoleKeyInfo('\0', ConsoleKey.F9, false, false, false);
            if (sequence == $"{ESC}[21~") return new ConsoleKeyInfo('\0', ConsoleKey.F10, false, false, false);
            if (sequence == $"{ESC}[23~") return new ConsoleKeyInfo('\0', ConsoleKey.F11, false, false, false);
            if (sequence == $"{ESC}[24~") return new ConsoleKeyInfo('\0', ConsoleKey.F12, false, false, false);
            
            // 處理 Home/End/Insert/Delete 等
            if (sequence == $"{ESC}[H") return new ConsoleKeyInfo('\0', ConsoleKey.Home, false, false, false);
            if (sequence == $"{ESC}[F") return new ConsoleKeyInfo('\0', ConsoleKey.End, false, false, false);
            if (sequence == $"{ESC}[2~") return new ConsoleKeyInfo('\0', ConsoleKey.Insert, false, false, false);
            if (sequence == $"{ESC}[3~") return new ConsoleKeyInfo('\0', ConsoleKey.Delete, false, false, false);
            if (sequence == $"{ESC}[5~") return new ConsoleKeyInfo('\0', ConsoleKey.PageUp, false, false, false);
            if (sequence == $"{ESC}[6~") return new ConsoleKeyInfo('\0', ConsoleKey.PageDown, false, false, false);
            
            // 檢測Alt組合鍵（通常是ESC後跟一個字符）
            if (sequence.Length == 2)
            {
                char c = sequence[1];
                if (c >= 'a' && c <= 'z')
                {
                    // Alt+字母
                    ConsoleKey key = (ConsoleKey)((int)ConsoleKey.A + (c - 'a'));
                    return new ConsoleKeyInfo(c, key, false, false, true);
                }
                else if (c >= '0' && c <= '9')
                {
                    // Alt+數字
                    ConsoleKey key = (ConsoleKey)((int)ConsoleKey.D0 + (c - '0'));
                    return new ConsoleKeyInfo(c, key, false, false, true);
                }
            }
            
            // 未識別的序列，返回ESC鍵
            return keyInfo;
        }
        
        // 檢測是否是中文或其他多字節字符
        // 這是通過檢查字符是否是高代理項或字符的編碼是否大於127
        if (keyInfo.KeyChar > 127 || char.IsHighSurrogate(keyInfo.KeyChar))
        {
            inputBuffer.Append(keyInfo.KeyChar);
            
            // 讀取額外的字節（如果有）
            int waitCount = 0;
            while (Console.KeyAvailable && waitCount < 10) // 最多等待10次以避免無限循環
            {
                var nextKeyInfo = Console.ReadKey(true);
                inputBuffer.Append(nextKeyInfo.KeyChar);
                waitCount++;
                
                // 如果是低代理項，表示我們已經讀取了完整的代理對
                if (char.IsLowSurrogate(nextKeyInfo.KeyChar))
                    break;
            }
            
            // 獲取完整的字符（可能是中文）
            string fullChar = inputBuffer.ToString();
            
            // 因為ConsoleKeyInfo不能直接存儲中文，所以我們使用空字符作為KeyChar，
            // 並將完整字符存儲在VimEditor的共享緩衝區中（這需要在VimEditor中實現）
            
            // 這裡我們製造一個特殊的ConsoleKeyInfo
            // 使用一個自定義的ConsoleKey值來表示這是一個中文字符
            // 當VimEditor讀取到這個按鍵時，需要特殊處理
            return new ConsoleKeyInfo(fullChar[0], ConsoleKey.Packet, false, false, false);
        }
        
        // 檢測CTRL和SHIFT組合鍵
        bool isCtrl = (keyInfo.Modifiers & ConsoleModifiers.Control) != 0;
        bool isShift = (keyInfo.Modifiers & ConsoleModifiers.Shift) != 0;
        
        // 返回原始按鍵信息
        return keyInfo;
    }
} 