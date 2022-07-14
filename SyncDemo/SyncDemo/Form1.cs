namespace SyncDemo
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

		private void button1_Click(object sender, EventArgs e)
		{
			//GetContentAsync();
			//AsyncCallApiAsyncAwait();
			//AsyncAwaitCallApiAsyncAwait();
			//AsyncCallStaticApiAsyncAwait();

			var task1 = Task.Run(AsyncCallApiAsync);
			var task2 = Task.Run(AsyncCallApiAsync);
			Task.WaitAll(task1, task2);
			Message.Text = "Complete";
		}

		private async Task AsyncCallApiAsync()
		{
			Message.Text = "Starting";
			var content = await GetContentAsync();
			Message.Text = "Received " + content;
		}

		private async Task AsyncCallApiAsyncAwait()
		{
			Message.Text = "Starting";
			var content = await GetContentAsyncAwait();
			Message.Text = "Received " + content;
		}

		private async Task AsyncAwaitCallApiAsyncAwait()
		{
			Message.Text = "Starting";
			var content = await GetContentAsyncAwait().ConfigureAwait(false);
			Message.Text = "Received " + content;
		}

		private async Task AsyncCallStaticApiAsyncAwait()
		{
			Message.Text = "Starting";
			var content = await StaticGetContentAsyncAwait();
			Message.Text = "Received " + content;
		}

		public async Task<string> GetContentAsync()
		{
			var httpClient = new HttpClient();
			var resp = await httpClient.GetAsync("https://reqres.in/api/users?page=2");
			var content = await resp.Content.ReadAsStringAsync();
			return content;
		}

		public async Task<string> GetContentAsyncAwait()
		{
			var httpClient = new HttpClient();
			var resp = await httpClient.GetAsync("https://reqres.in/api/users?page=2").ConfigureAwait(false);
			var content = await resp.Content.ReadAsStringAsync();
			return content;
		}

		public static async Task<string> StaticGetContentAsyncAwait()
		{
			var httpClient = new HttpClient();
			var resp = await httpClient.GetAsync("https://reqres.in/api/users?page=2").ConfigureAwait(false);
			var content = await resp.Content.ReadAsStringAsync();
			return content;
		}
	}
}