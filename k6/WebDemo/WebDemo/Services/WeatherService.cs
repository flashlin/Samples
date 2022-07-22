namespace WebDemo.Services
{
	public interface IWeatherService
	{
		Task<CityInfo> GetInfoAsync(string city);
	}

	public class WeatherService : IWeatherService
	{
		public async Task<CityInfo> GetInfoAsync(string city)
		{
			await Task.Delay(TimeSpan.FromSeconds(3));
			return new CityInfo()
			{
				Name = city,
				Temperature = new Random().Next(50)+1,
			};
		}
	}

	public class CityInfo
	{
		public string Name { get; set; }
		public int Temperature { get; set; }
	}
}
