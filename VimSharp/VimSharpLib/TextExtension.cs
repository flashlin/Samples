namespace VimSharpLib;
using System.Text;

/// <summary>
/// 提供文本處理的擴展方法
/// </summary>
public static class TextExtension
{
    
    /// <summary>
    /// 檢查字符是否為中文字符（在 Big5 編碼中佔用 2 個字節）
    /// </summary>
    /// <param name="c">要檢查的字符</param>
    /// <returns>如果是中文字符則返回 true，否則返回 false</returns>
    public static bool IsChinese(this char c)
    {
        // 使用 ASCII 編碼檢查字符的字節長度
        var bytes = Encoding.ASCII.GetBytes(new[] { c });
        return bytes.Length > 1;
    }
    
    /// <summary>
    /// 獲取字符在 Big5 編碼中的寬度（中文為 2，其他為 1）
    /// </summary>
    /// <param name="c">要檢查的字符</param>
    /// <returns>字符的顯示寬度</returns>
    public static int GetCharWidth(this char c)
    {
        return c.IsChinese() ? 2 : 1;
    }
    
    /// <summary>
    /// 計算字符串中的顯示寬度（考慮中文字符佔 2 個位置）
    /// </summary>
    /// <param name="text">要計算的字符串</param>
    /// <returns>字符串的顯示寬度</returns>
    public static int GetStringDisplayWidth(this string text)
    {
        int width = 0;
        foreach (char c in text)
        {
            width += c.GetCharWidth();
        }
        return width;
    }
    
    /// <summary>
    /// 根據顯示位置獲取字符串中的實際索引
    /// </summary>
    /// <param name="text">要處理的字符串</param>
    /// <param name="displayPosition">顯示位置</param>
    /// <returns>對應的字符索引</returns>
    public static int GetStringIndexFromDisplayPosition(this string text, int displayPosition)
    {
        int currentWidth = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (currentWidth >= displayPosition)
                return i;
                
            currentWidth += text[i].GetCharWidth();
        }
        return text.Length;
    }
} 