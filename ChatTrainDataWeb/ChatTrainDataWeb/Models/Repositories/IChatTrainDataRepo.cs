using ChatTrainDataWeb.Models.Entities;

namespace ChatTrainDataWeb.Models.Repositories;

public interface IChatTrainDataRepo
{
    void AddData(AddTrainDataDto data);
    List<TrainDataEntity> GetDataPage(int startId, int pageSize);
    void UpdateData(UpdateTrainDataDto data);
}