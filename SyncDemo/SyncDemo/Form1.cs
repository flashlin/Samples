using T1.Standard.Threads;

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
			//GetApiAsync();
			//AsyncCallApiAsyncAwait();
			//AsyncAwaitCallApiAsyncAwait();
			//AsyncCallStaticApiAsyncAwait();
			//InnerApiAsync();
			CallApiAsync();

			//var task1 = Task.Run(AsyncCallApiAsync);
			//var task2 = Task.Run(AsyncCallApiAsync);
			//Task.WaitAll(task1, task2);
			//Message.Text = "Complete";
		}

		private void CallApiAsync()
		{
			Message.Text = "Starting";
			var content = GetContentAsync().Result;
			Message.Text = "Received " + content;
		}

		private async Task<string> GetContentAsync()
		{
			return await GetApiAsync();
		}

		private async Task AsyncCallApiAsync()
		{
			Message.Text = "Starting";
			var content = await GetApiAsync();
			Message.Text = "Received " + content;
		}

		private async Task InnerApiAsync()
		{
			Message.Text = "Starting";
			var content = await InnerGetContentAsync();
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

		public async Task<string> GetApiAsync()
		{
			var httpClient = new HttpClient();
			var resp = await httpClient.GetAsync("https://reqres.in/api/users?page=2");
			var content = await resp.Content.ReadAsStringAsync();
			return content;
		}

		public Task<string> InnerGetContentAsync()
		{
			return GetContentAsyncAwait();
		}

		public async Task<string> GetContentAsyncAwait()
		{
			var httpClient = new HttpClient();
			var resp = await httpClient.GetAsync("https://reqres.in/api/users?page=2")
				.ConfigureAwait(false);
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