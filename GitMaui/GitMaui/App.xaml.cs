namespace GitMaui;

public partial class App : Application
{
	public App()
	{
		InitializeComponent();

		MainPage = new AppShell();
	}

	protected override Window CreateWindow(IActivationState activationState)
	{
		var window = base.CreateWindow(activationState);
		window.Created += Window_Created;
		//Created
		//Activted
		//Deactivated
		//Stopped
		//Resumed
		//Destroying
		return window;
	}

	private void Window_Created(object sender, EventArgs e)
	{
		//throw new NotImplementedException();
	}
}
