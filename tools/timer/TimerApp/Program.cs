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

        int minutes = 20; // 默認20分鐘
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-t" && i + 1 < args.Length)
            {
                if (int.TryParse(args[i + 1], out int parsedMinutes))
                {
                    minutes = parsedMinutes;
                    break;
                }
            }
        }

        Application.Run(new Form1(minutes));
    }    
}