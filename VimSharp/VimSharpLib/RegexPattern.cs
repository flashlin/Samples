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
    
    public bool IsMatch(List<ConsoleKeyInfo> keyBuffer)
    {
        if (keyBuffer.Count == 0)
            return false;
            
        // 將按鍵緩衝區轉換為字符串，用於正則匹配
        string input = ConvertKeyBufferToString(keyBuffer);
        
        // 使用正則表達式進行匹配
        return Regex.IsMatch(input, _pattern);
    }
    
    /// <summary>
    /// 將按鍵緩衝區轉換為字符串，用於正則匹配
    /// </summary>
    private string ConvertKeyBufferToString(List<ConsoleKeyInfo> keyBuffer)
    {
        var sb = new StringBuilder();
        
        foreach (var keyInfo in keyBuffer)
        {
            char keyChar = keyInfo.Key.ToChar();
            if (keyChar != '\0')
            {
                sb.Append(keyChar);
            }
        }
        
        return sb.ToString();
    }
} 