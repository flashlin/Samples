using ChatTrainDataWeb.Models.Entities;

namespace ChatTrainDataWeb.Models.Repositories;

public interface IChatTrainDataRepo
{
    void AddData(TrainDataDto data);
    List<TrainDataEntity> GetDataPage(int startId, int pageSize);
}