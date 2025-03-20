using System;
using System.Drawing;
using System.Windows.Forms;

namespace TimerApp;

public partial class Form1 : Form
{
    private int remainingSeconds;
    private Point lastLocation;
    private bool mouseDown;
    private Label timerLabel;

    public Form1()
    {
        InitializeComponent();

        // 設置視窗屬性
        this.FormBorderStyle = FormBorderStyle.None;
        this.BackColor = Color.Black;
        // 移除TransparencyKey設置，改用Opacity實現半透明
        // this.TransparencyKey = Color.Black;
        this.Opacity = 0.9; // 設置為90%不透明度（即10%透明）
        this.StartPosition = FormStartPosition.CenterScreen;
        this.TopMost = true;
        this.ShowInTaskbar = false;
        this.KeyPreview = true;

        // 創建倒數計時器標籤
        timerLabel = new Label();
        timerLabel.Text = "20:00";
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
        this.Size = new Size(labelSize.Width + 10, labelSize.Height + 10);
        
        // 設置標籤位置為視窗中央
        timerLabel.Location = new Point(5, 5);

        // 設置倒數計時器
        remainingSeconds = 20 * 60; // 20分鐘
        System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
        timer.Interval = 1000; // 1秒
        timer.Tick += (sender, e) => {
            remainingSeconds--;
            if (remainingSeconds >= 0)
            {
                int minutes = remainingSeconds / 60;
                int seconds = remainingSeconds % 60;
                timerLabel.Text = $"{minutes:D2}:{seconds:D2}";
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
