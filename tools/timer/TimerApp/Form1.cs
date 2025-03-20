using System;
using System.Drawing;
using System.Windows.Forms;

namespace TimerApp;

public partial class Form1 : Form
{
    private int remainingSeconds;
    private Point lastLocation;
    private bool mouseDown;

    public Form1()
    {
        InitializeComponent();

        // 設置視窗屬性
        this.FormBorderStyle = FormBorderStyle.None;
        this.BackColor = Color.Black;
        this.TransparencyKey = Color.Black;
        this.StartPosition = FormStartPosition.CenterScreen;
        this.TopMost = true;
        this.ShowInTaskbar = false;
        this.Size = new Size(200, 100);

        // 創建倒數計時器標籤
        Label timerLabel = new Label();
        timerLabel.Text = "20:00";
        timerLabel.Font = new Font("Arial", 24, FontStyle.Bold);
        timerLabel.ForeColor = Color.Yellow;
        timerLabel.BackColor = Color.Transparent;
        timerLabel.Dock = DockStyle.Fill;
        timerLabel.TextAlign = ContentAlignment.MiddleCenter;
        this.Controls.Add(timerLabel);

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
}
