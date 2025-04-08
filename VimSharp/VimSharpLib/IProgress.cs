namespace VimSharpLib;

/// <summary>
/// 進度報告介面
/// </summary>
public interface IProgress
{
    /// <summary>
    /// 顯示進度訊息
    /// </summary>
    /// <param name="message">進度訊息</param>
    void ShowProgress(string message);
} 