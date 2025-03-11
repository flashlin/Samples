namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;
using System.Text;

/// <summary>
/// 使用正則表達式模式匹配按鍵序列的類別
/// </summary>
public class PatternKeyPattern : IKeyPattern
{
    private readonly string _pattern;
    private readonly Regex _regex;

    /// <summary>
    /// 建構函數，接受正則表達式模式字串
    /// </summary>
    /// <param name="pattern">用於匹配按鍵序列的正則表達式模式</param>
    public PatternKeyPattern(string pattern)
    {
        _pattern = pattern ?? throw new ArgumentNullException(nameof(pattern));
        _regex = new Regex(pattern, RegexOptions.Compiled);
    }

    /// <summary>
    /// 檢查按鍵緩衝區是否匹配正則表達式模式
    /// </summary>
    /// <param name="keyBuffer">要檢查的按鍵緩衝區</param>
    /// <returns>如果匹配則返回 true，否則返回 false</returns>
    public bool IsMatch(List<ConsoleKey> keyBuffer)
    {
        if (keyBuffer == null || keyBuffer.Count == 0)
            return false;

        // 將按鍵緩衝區轉換為字符串
        string keyString = ConvertKeyBufferToString(keyBuffer);

        // 使用正則表達式進行匹配
        return _regex.IsMatch(keyString);
    }

    /// <summary>
    /// 將按鍵緩衝區轉換為字符串
    /// </summary>
    /// <param name="keyBuffer">要轉換的按鍵緩衝區</param>
    /// <returns>表示按鍵序列的字符串</returns>
    private string ConvertKeyBufferToString(List<ConsoleKey> keyBuffer)
    {
        var sb = new StringBuilder();
        
        foreach (var key in keyBuffer)
        {
            // 將 ConsoleKey 轉換為字符串表示
            string keyString = key.ToString();
            
            // 對於特殊按鍵，可以使用特定的表示方式
            // 例如，將方向鍵轉換為特定的字符
            switch (key)
            {
                case ConsoleKey.LeftArrow:
                    sb.Append("←");
                    break;
                case ConsoleKey.RightArrow:
                    sb.Append("→");
                    break;
                case ConsoleKey.UpArrow:
                    sb.Append("↑");
                    break;
                case ConsoleKey.DownArrow:
                    sb.Append("↓");
                    break;
                case ConsoleKey.Enter:
                    sb.Append("↵");
                    break;
                case ConsoleKey.Escape:
                    sb.Append("Esc");
                    break;
                case ConsoleKey.Tab:
                    sb.Append("Tab");
                    break;
                case ConsoleKey.Backspace:
                    sb.Append("⌫");
                    break;
                case ConsoleKey.Delete:
                    sb.Append("Del");
                    break;
                case ConsoleKey.Home:
                    sb.Append("Home");
                    break;
                case ConsoleKey.End:
                    sb.Append("End");
                    break;
                case ConsoleKey.PageUp:
                    sb.Append("PgUp");
                    break;
                case ConsoleKey.PageDown:
                    sb.Append("PgDn");
                    break;
                case ConsoleKey.Insert:
                    sb.Append("Ins");
                    break;
                default:
                    // 對於字母和數字按鍵，直接使用其字符表示
                    if (key >= ConsoleKey.A && key <= ConsoleKey.Z)
                    {
                        sb.Append((char)('A' + (key - ConsoleKey.A)));
                    }
                    else if (key >= ConsoleKey.D0 && key <= ConsoleKey.D9)
                    {
                        sb.Append((char)('0' + (key - ConsoleKey.D0)));
                    }
                    else
                    {
                        // 對於其他按鍵，使用其名稱
                        sb.Append(keyString);
                    }
                    break;
            }
        }
        
        return sb.ToString();
    }
} 