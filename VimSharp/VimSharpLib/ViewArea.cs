namespace VimSharpLib;

/// <summary>
/// 表示控制台中的矩形區域
/// </summary>
public class ViewArea
{
    /// <summary>
    /// 矩形左上角的 X 座標
    /// </summary>
    public int X { get; set; }
    
    /// <summary>
    /// 矩形左上角的 Y 座標
    /// </summary>
    public int Y { get; set; }
    
    /// <summary>
    /// 矩形的寬度
    /// </summary>
    public int Width { get; set; }
    
    /// <summary>
    /// 矩形的高度
    /// </summary>
    public int Height { get; set; }
    
    public int Right => X + Width - 1;
    public int Bottom => Y + Height - 1;
    
    /// <summary>
    /// 建立一個新的 ConsoleRectangle 實例
    /// </summary>
    public ViewArea()
    {
    }
    
    /// <summary>
    /// 建立一個新的 ConsoleRectangle 實例，並指定座標和尺寸
    /// </summary>
    /// <param name="x">矩形左上角的 X 座標</param>
    /// <param name="y">矩形左上角的 Y 座標</param>
    /// <param name="width">矩形的寬度</param>
    /// <param name="height">矩形的高度</param>
    public ViewArea(int x, int y, int width, int height)
    {
        X = x;
        Y = y;
        Width = width;
        Height = height;
    }
    
    /// <summary>
    /// 檢查指定的點是否在矩形內
    /// </summary>
    /// <param name="x">點的 X 座標</param>
    /// <param name="y">點的 Y 座標</param>
    /// <returns>如果點在矩形內，則為 true；否則為 false</returns>
    public bool Contains(int x, int y)
    {
        return x >= X && x < X + Width && y >= Y && y < Y + Height;
    }
    
    /// <summary>
    /// 檢查此矩形是否與另一個矩形相交
    /// </summary>
    /// <param name="other">要檢查的另一個矩形</param>
    /// <returns>如果矩形相交，則為 true；否則為 false</returns>
    public bool Intersects(ViewArea other)
    {
        return X < other.X + other.Width && X + Width > other.X &&
               Y < other.Y + other.Height && Y + Height > other.Y;
    }
} 