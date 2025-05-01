using System.Runtime.InteropServices;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace ScreenCap;

public partial class Form1 : Form
{
    [DllImport("user32.dll")]
    private static extern bool RegisterHotKey(IntPtr hWnd, int id, int fsModifiers, int vk);

    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

    private const int CTRL = 0x0002;
    private const int F12_KEY = 0x7B;
    private const int HOTKEY_ID = 1;

    private bool isCapturing = false;
    private bool hasStartPoint = false;  // 新增：用於追蹤是否已有起始點
    private Point startPoint;
    private Point endPoint;
    private Form overlayForm;
    private PictureBox previewBox;
    private Point currentMousePosition;

    public Form1()
    {
        InitializeComponent();
        InitializePreviewBox();
        RegisterGlobalHotKey();
        this.FormClosing += Form1_FormClosing;
        this.KeyPreview = true; // 允許表單攔截鍵盤事件
        this.KeyDown += Form1_KeyDown;
    }

    private void InitializePreviewBox()
    {
        previewBox = new PictureBox
        {
            SizeMode = PictureBoxSizeMode.AutoSize,
            Location = new Point(12, 12),
        };
        this.Controls.Add(previewBox);
    }

    private void RegisterGlobalHotKey()
    {
        RegisterHotKey(this.Handle, HOTKEY_ID, CTRL, F12_KEY);
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == 0x0312 && m.WParam.ToInt32() == HOTKEY_ID)
        {
            StartCapture();
        }
        base.WndProc(ref m);
    }

    private void StartCapture()
    {
        if (!isCapturing)
        {
            isCapturing = true;
            hasStartPoint = false;
            startPoint = Point.Empty;
            endPoint = Point.Empty;
            CreateOverlayForm();
        }
    }

    private void CreateOverlayForm()
    {
        overlayForm = new Form
        {
            FormBorderStyle = FormBorderStyle.None,
            ShowInTaskbar = false,
            TopMost = true,
            WindowState = FormWindowState.Maximized,
            BackColor = Color.FromArgb(10, 10, 10),
            Opacity = 0.3,
            Cursor = Cursors.Cross
        };

        overlayForm.MouseDown += OverlayForm_MouseDown;
        overlayForm.MouseMove += OverlayForm_MouseMove;
        overlayForm.Paint += OverlayForm_Paint;

        overlayForm.Show();
    }

    private void OverlayForm_MouseDown(object sender, MouseEventArgs e)
    {
        if (e.Button == MouseButtons.Left)
        {
            if (!hasStartPoint)
            {
                // 第一次點擊：設置起始點
                hasStartPoint = true;
                startPoint = e.Location;
                endPoint = startPoint;
            }
            else
            {
                // 第二次點擊：設置終點並完成截圖
                endPoint = e.Location;
                CaptureScreen();
                isCapturing = false;
                hasStartPoint = false;
                overlayForm.Close();
            }
            overlayForm.Refresh();
        }
    }

    private void OverlayForm_MouseMove(object sender, MouseEventArgs e)
    {
        currentMousePosition = e.Location;
        if (hasStartPoint)
        {
            endPoint = e.Location;
        }
        overlayForm.Refresh();
    }

    private void OverlayForm_Paint(object sender, PaintEventArgs e)
    {
        using (Pen crosshairPen = new Pen(Color.White, 1))
        {
            crosshairPen.DashStyle = DashStyle.Dash;
            
            // 繪製十字線
            e.Graphics.DrawLine(crosshairPen, new Point(0, currentMousePosition.Y), 
                new Point(overlayForm.Width, currentMousePosition.Y));
            e.Graphics.DrawLine(crosshairPen, new Point(currentMousePosition.X, 0), 
                new Point(currentMousePosition.X, overlayForm.Height));

            if (hasStartPoint)
            {
                var rect = GetCaptureRect();
                using (Pen selectionPen = new Pen(Color.LightGreen, 2))
                {
                    selectionPen.DashStyle = DashStyle.Dash;
                    e.Graphics.DrawRectangle(selectionPen, rect);

                    // 顯示尺寸資訊
                    string dimensions = $"{rect.Width} x {rect.Height}";
                    using (Font font = new Font("Arial", 10))
                    {
                        // 計算文字大小
                        var textSize = e.Graphics.MeasureString(dimensions, font);
                        var textRect = new Rectangle(
                            rect.X,
                            rect.Y - 25,
                            (int)textSize.Width + 10,
                            (int)textSize.Height + 6
                        );

                        // 繪製黑色背景
                        using (var brush = new SolidBrush(Color.FromArgb(200, 0, 0, 0)))
                        {
                            e.Graphics.FillRectangle(brush, textRect);
                        }

                        // 繪製白色文字
                        using (var brush = new SolidBrush(Color.White))
                        {
                            e.Graphics.DrawString(dimensions, font, brush, 
                                textRect.X + 5, textRect.Y + 3);
                        }
                    }
                }
            }
        }
    }

    private Rectangle GetCaptureRect()
    {
        return new Rectangle(
            Math.Min(startPoint.X, endPoint.X),
            Math.Min(startPoint.Y, endPoint.Y),
            Math.Abs(endPoint.X - startPoint.X),
            Math.Abs(endPoint.Y - startPoint.Y));
    }

    private void CaptureScreen()
    {
        var rect = GetCaptureRect();
        if (rect.Width <= 0 || rect.Height <= 0) return;

        using (Bitmap bitmap = new Bitmap(rect.Width, rect.Height))
        {
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                g.CopyFromScreen(
                    rect.Left + overlayForm.Left,
                    rect.Top + overlayForm.Top,
                    0, 0, rect.Size);
            }

            // 將圖片複製到剪貼簿
            Clipboard.SetImage(bitmap);
            
            // 顯示在 PictureBox 中
            previewBox.Image?.Dispose();
            previewBox.Image = new Bitmap(bitmap);
            this.ClientSize = new Size(
                Math.Min(800, bitmap.Width + 24),
                Math.Min(600, bitmap.Height + 24));
        }
    }

    private void Form1_FormClosing(object sender, FormClosingEventArgs e)
    {
        UnregisterHotKey(this.Handle, HOTKEY_ID);
    }

    private void Form1_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Control && e.KeyCode == Keys.S)
        {
            SaveImage();
            e.Handled = true;
        }
        else if (e.Control && e.KeyCode == Keys.V)
        {
            PasteImage();
            e.Handled = true;
        }
    }

    private void SaveImage()
    {
        if (previewBox.Image == null)
        {
            MessageBox.Show("沒有可儲存的截圖！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }
        using (SaveFileDialog sfd = new SaveFileDialog())
        {
            sfd.Filter = "PNG 圖片 (*.png)|*.png";
            sfd.DefaultExt = "png";
            sfd.AddExtension = true;
            sfd.Title = "儲存截圖";
            if (sfd.ShowDialog() == DialogResult.OK)
            {
                previewBox.Image.Save(sfd.FileName, System.Drawing.Imaging.ImageFormat.Png);
            }
        }
    }

    private void PasteImage()
    {
        try
        {
            if (Clipboard.ContainsImage())
            {
                using (Image clipboardImage = Clipboard.GetImage())
                {
                    if (clipboardImage != null)
                    {
                        // 釋放舊的圖片資源
                        previewBox.Image?.Dispose();
                        
                        // 建立新的 Bitmap 並複製剪貼簿圖片
                        previewBox.Image = new Bitmap(clipboardImage);
                        
                        // 調整視窗大小以適應圖片
                        this.ClientSize = new Size(
                            Math.Min(800, previewBox.Image.Width + 24),
                            Math.Min(600, previewBox.Image.Height + 24));
                    }
                }
            }
            else
            {
                MessageBox.Show("剪貼簿中沒有圖片！", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"無法貼上圖片：{ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
