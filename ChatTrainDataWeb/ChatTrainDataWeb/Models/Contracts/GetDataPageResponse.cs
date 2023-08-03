namespace ChatTrainDataWeb.Models.Contracts;

public class GetDataPageResponse
{
    public List<TrainDataItem> Items { get; set; } = new();
}