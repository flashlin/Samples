using System;
using System.Drawing;
using System.Windows.Forms;
using System.Runtime.InteropServices;

namespace TimerApp;

public partial class Form1 : Form
{
    private int remainingSeconds;
    private Point lastLocation;
    private bool mouseDown;
    private Label timerLabel;
    private Panel dragHandle; // 新增拖曳把手

    // Win32 API 引入
    [DllImport("user32.dll")]
    private static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);
    
    [DllImport("user32.dll")]
    private static extern int GetWindowLong(IntPtr hWnd, int nIndex);
    
    // 窗口樣式常量
    private const int GWL_EXSTYLE = -20;
    private const int WS_EX_LAYERED = 0x80000;
    private const int WS_EX_TRANSPARENT = 0x20;

    public Form1(int minutes = 20)
    {
        InitializeComponent();

        // 設置視窗屬性
        this.FormBorderStyle = FormBorderStyle.None;
        this.BackColor = Color.Black;
        this.TransparencyKey = Color.Black; // 設置背景透明
        this.Opacity = 0.9; // 設置為90%不透明度，這將影響標籤
        this.StartPosition = FormStartPosition.CenterScreen;
        this.TopMost = true;
        this.ShowInTaskbar = false;
        this.KeyPreview = true;

        // 創建倒數計時器標籤
        timerLabel = new Label();
        timerLabel.Text = $"{minutes:D2}:00";
        timerLabel.Font = new Font("Arial", 24, FontStyle.Bold);
        timerLabel.ForeColor = Color.Yellow;
        timerLabel.BackColor = Color.Transparent;
        timerLabel.AutoSize = true;
        timerLabel.TextAlign = ContentAlignment.MiddleCenter;
        
        // 添加標籤的滑鼠事件處理
        timerLabel.MouseDown += Form1_MouseDown;
        timerLabel.MouseMove += Form1_MouseMove;
        timerLabel.MouseUp += Form1_MouseUp;
        timerLabel.MouseEnter += Form1_MouseEnter;
        timerLabel.MouseLeave += Form1_MouseLeave;
        
        this.Controls.Add(timerLabel);
        
        // 計算並設置視窗大小
        Size labelSize = TextRenderer.MeasureText(timerLabel.Text, timerLabel.Font);
        this.Size = new Size(labelSize.Width + 20, labelSize.Height + 20);
        
        // 設置標籤位置為視窗中央
        timerLabel.Location = new Point(10, 10);
        
        // 創建拖曳把手（小方塊）
        dragHandle = new Panel();
        dragHandle.Size = new Size(10, 10);
        dragHandle.BackColor = Color.Yellow;
        dragHandle.Location = new Point(timerLabel.Location.X + labelSize.Width + 5, timerLabel.Location.Y - 5);
        dragHandle.Cursor = Cursors.SizeAll;
        
        // 設置拖曳把手的滑鼠事件
        dragHandle.MouseDown += Form1_MouseDown;
        dragHandle.MouseMove += Form1_MouseMove;
        dragHandle.MouseUp += Form1_MouseUp;
        
        this.Controls.Add(dragHandle);

        // 設置倒數計時器
        remainingSeconds = minutes * 60; // 使用傳入的分鐘數
        System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
        timer.Interval = 1000; // 1秒
        timer.Tick += (sender, e) => {
            remainingSeconds--;
            if (remainingSeconds >= 0)
            {
                int mins = remainingSeconds / 60;
                int secs = remainingSeconds % 60;
                timerLabel.Text = $"{mins:D2}:{secs:D2}";
                
                // 更新拖曳把手位置（如果標籤大小變化）
                Size newLabelSize = TextRenderer.MeasureText(timerLabel.Text, timerLabel.Font);
                dragHandle.Location = new Point(timerLabel.Location.X + newLabelSize.Width, timerLabel.Location.Y);
            }
            else
            {
                timer.Stop();
            }
        };
        timer.Start();

        // 添加滑鼠事件處理
        this.MouseDown += Form1_MouseDown;
        this.MouseMove += Form1_MouseMove;
        this.MouseUp += Form1_MouseUp;
        this.MouseEnter += Form1_MouseEnter;
        this.MouseLeave += Form1_MouseLeave;
        
        // 添加鍵盤事件處理
        this.KeyDown += Form1_KeyDown;

        // 使窗口允許點擊穿透但保持標籤可點擊
        this.Load += (s, e) => {
            // 設置窗口為分層窗口
            int exStyle = GetWindowLong(this.Handle, GWL_EXSTYLE);
            exStyle |= WS_EX_LAYERED;
            // 不設置 WS_EX_TRANSPARENT 使窗口可以接收滑鼠事件
            SetWindowLong(this.Handle, GWL_EXSTYLE, exStyle);
        };
    }

    private void Form1_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Escape)
        {
            this.Close();
        }
    }

    private void Form1_MouseDown(object? sender, MouseEventArgs e)
    {
        mouseDown = true;
        lastLocation = e.Location;
    }

    private void Form1_MouseMove(object? sender, MouseEventArgs e)
    {
        if (mouseDown)
        {
            this.Location = new Point(
                (this.Location.X - lastLocation.X) + e.X,
                (this.Location.Y - lastLocation.Y) + e.Y);

            this.Update();
        }
    }

    private void Form1_MouseUp(object? sender, MouseEventArgs e)
    {
        mouseDown = false;
    }

    private void Form1_MouseEnter(object? sender, EventArgs e)
    {
        this.Cursor = Cursors.SizeAll;
    }

    private void Form1_MouseLeave(object? sender, EventArgs e)
    {
        this.Cursor = Cursors.Default;
    }
}
