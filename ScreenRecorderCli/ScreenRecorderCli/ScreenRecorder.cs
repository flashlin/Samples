using ScreenRecorderLib;

public class ScreenOption
{
	public static RecordingSourceBase MainMonitor = new DisplayRecordingSource(DisplayRecordingSource.MainMonitor);
	public static RecordingSourceBase Monitor2 = new DisplayRecordingSource(@"\\.\DISPLAY2");
}

public class ScreenRecorder
{
	Recorder _rec;

	public string VideoPath { get; set; } = "d:/demo";

	public RecordingSourceBase RecordingSource { get; set; } = ScreenOption.MainMonitor;

	public void Start()
	{
		var videoFile = VideoPath + "/" + GetTempFile();

		var options = new RecorderOptions
		{
			SourceOptions = new SourceOptions
			{
				RecordingSources = new List<RecordingSourceBase>()
				{
					//new DisplayRecordingSource(@"\\.\DISPLAY2")
					//new DisplayRecordingSource(DisplayRecordingSource.MainMonitor)
					RecordingSource
				}
			},
			OutputOptions = new OutputOptions
			{
				RecorderMode = RecorderMode.Video,
				OutputFrameSize = new ScreenSize(1280, 720),
				//Stretch controls how the resizing is done, if the new aspect ratio differs.
				Stretch = StretchMode.Uniform,
				//SourceRect = new ScreenRect(100, 100, 500, 500)
			},
			AudioOptions = new AudioOptions
			{
				Bitrate = AudioBitrate.bitrate_128kbps,
				Channels = AudioChannels.Stereo,
				IsAudioEnabled = true,
			},
			VideoEncoderOptions = new VideoEncoderOptions
			{
				Bitrate = 8000 * 1000,
				Framerate = 15,
				IsFixedFramerate = true,
				Encoder = new H264VideoEncoder
				{
					BitrateMode = H264BitrateControlMode.CBR,
					EncoderProfile = H264Profile.Main,
				},
				IsFragmentedMp4Enabled = true,
				IsThrottlingDisabled = false,
				IsHardwareEncodingEnabled = true,
				IsLowLatencyEnabled = false,
				IsMp4FastStartEnabled = true
			},
		};

		_rec = Recorder.CreateRecorder(options);
		_rec.OnRecordingComplete += OnRecordingComplete;
		_rec.OnRecordingFailed += OnRecordingFailed;
		_rec.OnStatusChanged += OnStatusChanged;
		_rec.Record(videoFile);
	}

	public void Stop()
	{
		_rec.Stop();
	}

	private string GetTempFile()
	{
		return $"{DateTime.Now.ToString("yyyyMMdd_HHmmss")}.mp4";
	}
	private void OnRecordingComplete(object? sender, RecordingCompleteEventArgs e)
	{
		var path = e.FilePath;
	}

	private void OnRecordingFailed(object? sender, RecordingFailedEventArgs e)
	{
		var error = e.Error;
	}

	private void OnStatusChanged(object? sender, RecordingStatusEventArgs e)
	{
		var status = e.Status;
	}
}