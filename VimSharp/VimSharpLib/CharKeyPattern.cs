namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Linq;

public class CharKeyPattern : IKeyPattern
{
    private readonly char _char;

    public CharKeyPattern(char c)
    {
        _char = c;
    }

    public bool IsMatch(List<ConsoleKeyInfo> keyBuffer)
    {
        if (keyBuffer.Count == 0)
            return false;

        // 檢查按鍵緩衝區的最後一個按鍵是否對應於指定的字符
        var lastKey = keyBuffer.Last().Key;
        
        // 將 ConsoleKey 轉換為字符
        char keyChar;
        
        // 處理特殊符號
        if (_char == '$' && lastKey == ConsoleKey.D4)
        {
            return true;
        }
        else if (_char == '^' && lastKey == ConsoleKey.D6)
        {
            return true;
        }
        else
        {
            keyChar = GetCharFromConsoleKey(lastKey);
            return keyChar == _char;
        }
    }
    
    private char GetCharFromConsoleKey(ConsoleKey key)
    {
        // 處理特殊字符
        switch (key)
        {
            case ConsoleKey.D0: return '0';
            case ConsoleKey.D1: return '1';
            case ConsoleKey.D2: return '2';
            case ConsoleKey.D3: return '3';
            case ConsoleKey.D4: return '4';
            case ConsoleKey.D5: return '5';
            case ConsoleKey.D6: return '6';
            case ConsoleKey.D7: return '7';
            case ConsoleKey.D8: return '8';
            case ConsoleKey.D9: return '9';
            case ConsoleKey.OemPlus: return '+';
            case ConsoleKey.OemMinus: return '-';
            case ConsoleKey.OemPeriod: return '.';
            case ConsoleKey.OemComma: return ',';
            case ConsoleKey.Oem1: return ';';
            case ConsoleKey.Oem2: return '/';
            case ConsoleKey.Oem3: return '`';
            case ConsoleKey.Oem4: return '[';
            case ConsoleKey.Oem5: return '\\';
            case ConsoleKey.Oem6: return ']';
            case ConsoleKey.Oem7: return '\'';
            case ConsoleKey.Oem8: return '!';
            case ConsoleKey.Divide: return '/';
            case ConsoleKey.Multiply: return '*';
            case ConsoleKey.Subtract: return '-';
            case ConsoleKey.Add: return '+';
            case ConsoleKey.Decimal: return '.';
            // 處理字母
            default:
                if (key >= ConsoleKey.A && key <= ConsoleKey.Z)
                {
                    return (char)('a' + (key - ConsoleKey.A));
                }
                // 處理特殊符號
                if (key == ConsoleKey.D4) // $ 符號 (Shift + 4)
                {
                    return '$';
                }
                if (key == ConsoleKey.D6) // ^ 符號 (Shift + 6)
                {
                    return '^';
                }
                return '\0';
        }
    }
} 