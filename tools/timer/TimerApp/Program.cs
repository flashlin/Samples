using System;
using System.Windows.Forms;

namespace TimerApp;

static class Program
{
    /// <summary>
    ///  The main entry point for the application.
    /// </summary>
    [STAThread]
    static void Main(string[] args)
    {
        // To customize application configuration such as set high DPI settings or default font,
        // see https://aka.ms/applicationconfiguration.
        Application.SetHighDpiMode(HighDpiMode.SystemAware);
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        string timeString = "20:00"; // 默認20分鐘
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-t" && i + 1 < args.Length)
            {
                // 驗證時間格式是否為 MM:SS
                string timeArg = args[i + 1];
                if (System.Text.RegularExpressions.Regex.IsMatch(timeArg, @"^\d{1,2}:\d{2}$"))
                {
                    timeString = timeArg;
                    break;
                }
            }
        }

        Application.Run(new Form1(timeString));
    }    
}