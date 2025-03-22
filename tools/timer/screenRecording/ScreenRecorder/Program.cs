using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Windows.Forms;
using FFmpeg.AutoGen;
using NAudio.Wave;

class Program
{
    public static bool isRecording = false;
    private static bool isRunning = true;
    public static Rectangle displayRect;
    private static int deviceNumber = 0;
    private static Form overlayForm;
    private static Process ffmpegProcess;
    public const int BORDER_WIDTH = 7;  // 邊框粗細

    [DllImport("user32.dll")]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);

    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

    [STAThread]
    static void Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        if (args.Length < 4)
        {
            Console.WriteLine("使用方式: ScreenRecorder.exe x y width height [-d deviceNumber]");
            return;
        }

        // 解析參數
        int x = int.Parse(args[0]);
        int y = int.Parse(args[1]);
        int width = int.Parse(args[2]);
        int height = int.Parse(args[3]);

        // 檢查是否有指定螢幕編號
        for (int i = 4; i < args.Length; i++)
        {
            if (args[i] == "-d" && i + 1 < args.Length)
            {
                deviceNumber = int.Parse(args[i + 1]);
                break;
            }
        }

        displayRect = new Rectangle(x, y, width, height);

        // 建立透明視窗
        overlayForm = new Form
        {
            FormBorderStyle = FormBorderStyle.None,
            ShowInTaskbar = false,
            TopMost = true,
            BackColor = Color.White,
            TransparencyKey = Color.White,
            Size = new Size(width + BORDER_WIDTH * 2, height + BORDER_WIDTH * 2),  // 增加邊框寬度的兩倍
            Location = new Point(x - BORDER_WIDTH, y - BORDER_WIDTH),      // 向左上偏移邊框寬度
            Opacity = 1.0,
            StartPosition = FormStartPosition.Manual
        };

        // 註冊 F12 熱鍵
        RegisterHotKey(overlayForm.Handle, 1, 0, 0x7B); // 0x7B 是 F12 的虛擬鍵碼

        // 處理熱鍵事件
        overlayForm.KeyDown += (s, e) =>
        {
            if (e.KeyCode == Keys.F12)
            {
                ToggleRecording();
            }
        };

        // 處理視窗關閉事件
        overlayForm.FormClosing += (s, e) =>
        {
            isRunning = false;
            if (isRecording)
            {
                StopRecording();
            }
        };

        // 使用 Application.Run 來運行 Windows Forms 應用程式
        Application.Run(overlayForm);
    }

    private static void ToggleRecording()
    {
        if (!isRecording)
        {
            StartRecording();
        }
        else
        {
            StopRecording();
        }
    }

    public static void StartRecording()
    {
        isRecording = true;
        overlayForm.SetRecordingState(true);

        string outputFile = $"recording_{DateTime.Now:yyyyMMdd_HHmmss}.mp4";
        string ffmpegArgs = $"-f gdigrab -framerate 30 -offset_x {displayRect.X} -offset_y {displayRect.Y} -video_size {displayRect.Width}x{displayRect.Height} -i desktop -c:v libx264 -preset ultrafast -y {outputFile}";

        ffmpegProcess = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = @"D:\VDisk\GSoft\ffmpeg-4.4.1_2021-12-14\ffmpeg.exe",
                Arguments = ffmpegArgs,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardInput = true,
                CreateNoWindow = true
            }
        };

        ffmpegProcess.Start();
    }

    public static void StopRecording()
    {
        isRecording = false;
        overlayForm.SetRecordingState(false);

        if (ffmpegProcess != null && !ffmpegProcess.HasExited)
        {
            ffmpegProcess.StandardInput.WriteLine("q");
            ffmpegProcess.WaitForExit();
        }
    }

    public static void UpdateRecordingArea(int deltaX, int deltaY)
    {
        displayRect.X += deltaX;
        displayRect.Y += deltaY;
    }
}

// 用於顯示方框的透明視窗
public class Form : System.Windows.Forms.Form
{
    private bool isRecording = false;
    private Point lastPoint;

    public Form()
    {
        // 設定預設游標
        this.Cursor = Cursors.Default;
        // 設定邊距為 0
        this.Padding = new Padding(0);
        // 設定視窗樣式，移除標題列
        this.FormBorderStyle = FormBorderStyle.None;
    }

    public void SetRecordingState(bool recording)
    {
        isRecording = recording;
        this.Invalidate(); // 觸發重繪
    }

    protected override void OnMouseEnter(EventArgs e)
    {
        base.OnMouseEnter(e);
        this.Cursor = Cursors.SizeAll; // 當滑鼠移入時，改變為移動游標
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);
        this.Cursor = Cursors.Default; // 當滑鼠移出時，恢復預設游標
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        lastPoint = e.Location;
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (e.Button == MouseButtons.Left && !Program.isRecording)  // 只在非錄影狀態下允許拖曳
        {
            int deltaX = e.X - lastPoint.X;
            int deltaY = e.Y - lastPoint.Y;
            
            this.Left += deltaX;
            this.Top += deltaY;

            // 更新錄影範圍
            Program.displayRect.X += deltaX;
            Program.displayRect.Y += deltaY;
        }
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        using (Pen pen = new Pen(isRecording ? Color.Red : Color.Green, Program.BORDER_WIDTH))
        {
            // 從真正的視窗左上角開始繪製方框
            e.Graphics.DrawRectangle(pen, 0, 0, Width - 1, Height - 1);
        }
    }
}
