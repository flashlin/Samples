using System;

namespace Generated
{
    public class BatchResultHistoryEntity
    {
        public int ID { get; set; }
        public int BatchId { get; set; }
        public int? TotalBets { get; set; }
        public int? TotalSuccessBets { get; set; }
        public bool? StatusUpdated { get; set; }
        public required string Action { get; set; }
        public bool? IsSuccess { get; set; }
        public DateTime? StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string Remark { get; set; }
        public required string KafkaInfo { get; set; }
    }
}
