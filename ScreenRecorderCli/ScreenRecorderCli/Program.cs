Console.WriteLine("Hello, Screen Recorder!");
var screenRecorder = new ScreenRecorder();
screenRecorder.Start();

Console.WriteLine("Press ANY KEY TO EXIT!");
Console.ReadLine();
screenRecorder.Stop();