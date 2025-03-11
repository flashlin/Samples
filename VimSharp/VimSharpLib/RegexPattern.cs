namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Text;

public class RegexPattern : IKeyPattern
{
    private readonly string _pattern;
    
    public RegexPattern(string pattern)
    {
        _pattern = pattern;
    }
    
    public bool IsMatch(List<ConsoleKey> keyBuffer)
    {
        if (keyBuffer == null || keyBuffer.Count == 0)
            return false;
            
        // 將按鍵緩衝區轉換為字符串
        string input = ConvertKeyBufferToString(keyBuffer);
        
        // 使用正則表達式進行匹配
        return Regex.IsMatch(input, _pattern);
    }
    
    /// <summary>
    /// 將按鍵緩衝區轉換為字符串
    /// </summary>
    private string ConvertKeyBufferToString(List<ConsoleKey> keyBuffer)
    {
        var sb = new StringBuilder();
        
        foreach (var key in keyBuffer)
        {
            char keyChar = GetCharFromConsoleKey(key);
            if (keyChar != '\0')
            {
                sb.Append(keyChar);
            }
        }
        
        return sb.ToString();
    }
    
    /// <summary>
    /// 將ConsoleKey轉換為字符
    /// </summary>
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
            case ConsoleKey.J: return 'J';
            // 處理字母
            default:
                if (key >= ConsoleKey.A && key <= ConsoleKey.Z)
                {
                    return (char)('a' + (key - ConsoleKey.A));
                }
                return '\0';
        }
    }
} 